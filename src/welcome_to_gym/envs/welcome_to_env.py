from enum import IntEnum
from typing import Iterable, Optional, Sized
import numpy as np
from gym import (
    spaces,
    Env
)

MAX_HOUSE_NUMBER = 15
MAX_TEMP_NUMBER = MAX_HOUSE_NUMBER + 2


class ActionCard(IntEnum):
    TEMP = 0
    BIS = 1
    POOL = 2
    FENCE = 3
    PARK = 4
    ESTATE = 5
    NONE = 6


def pad_2d_array(arr: Iterable[Sized]) -> np.ndarray:
    max_len = max([len(x) for x in arr])
    return np.stack([np.pad(np.array(x), (0, max_len - len(x))) for x in arr])


class GlobalScores:
    def __init__(self, /, estate_scores: Iterable[Iterable[int]], bis_scores: Iterable[int], pool_scores: Iterable[int], failure_scores: Iterable[int]):
        self.ESTATE_SCORES = estate_scores
        self.BIS_SCORES = bis_scores
        self.POOL_SCORES = pool_scores
        self.FAILURE_SCORES = failure_scores
        self.NUM_ESTATE_TYPES = len(self.ESTATE_SCORES)
        self.MAX_ESTATE_COUNT = np.array([len(x)-1
                                          for x in self.ESTATE_SCORES], dtype=np.uint8)
        self.ESTATE_SCORES_ARRAY = pad_2d_array(self.ESTATE_SCORES)

        self.observation_space = {
            "bis_score": spaces.Discrete(len(self.BIS_SCORES)),
            "pool_score": spaces.Discrete(len(self.POOL_SCORES)),
            "estate_scores": spaces.MultiDiscrete([len(x) for x in self.ESTATE_SCORES]),
            "failure_score": spaces.Discrete(len(self.FAILURE_SCORES))
        }

    def reset(self):
        self.pool_score = 0
        self.bis_score = 0
        self.estate_scores = np.zeros(
            self.NUM_ESTATE_TYPES, dtype=np.uint8)
        self.failure_score = 0

    def get_observation(self):
        return {
            "bis_score": self.bis_score,
            "pool_score": self.pool_score,
            "estate_scores": self.estate_scores,
            "failure_score": self.failure_score
        }

    def advance_pool(self):
        if self.pool_score < len(self.POOL_SCORES) - 1:
            self.pool_score += 1

    def advance_bis(self):
        if self.bis_score < len(self.BIS_SCORES) - 1:
            self.bis_score += 1

    def advance_estate(self, estate_num: int):
        self.estate_scores[estate_num] += 1

    def advance_failure_score(self):
        self.failure_score += 1

    def is_max_failure_reached(self):
        return self.failure_score == len(self.FAILURE_SCORES) - 1

    def get_estate_mask(self) -> np.ndarray[bool]:
        return self.estate_scores < self.MAX_ESTATE_COUNT

    def get_pool_mask(self) -> bool:
        return self.pool_score < len(self.POOL_SCORES) - 1

    def get_estate_scores(self) -> list[int]:
        return self.ESTATE_SCORES_ARRAY[np.mgrid[:self.NUM_ESTATE_TYPES], self.estate_scores]

    def get_pool_score(self) -> int:
        return self.POOL_SCORES[self.pool_score]

    def get_bis_score(self) -> int:
        return self.BIS_SCORES[self.bis_score]

    def get_failure_score(self) -> int:
        return self.FAILURE_SCORES[self.failure_score]

    def get_global_score(self) -> int:
        return self.get_pool_score() + self.get_bis_score() + self.get_failure_score()


class Row:
    HOUSE_NUMBER_DIAG = np.pad(
        np.eye(MAX_TEMP_NUMBER, dtype=bool), ((1, 0), (0, 0)))

    TEMP_OFFSET = np.array([-2, -1, 1, 2])

    def __init__(self, /, size: int, pools: Iterable[int], parks: Iterable[int], global_scores: GlobalScores):
        self.SIZE = size
        self.POOL_IDX = tuple(pools)
        self.NUM_PARKS = len(parks)
        self.NUM_FENCES = self.SIZE - 1
        self.NUM_POOLS = len(self.POOL_IDX)
        self.PARK_SCORES = tuple(parks)

        self.global_scores = global_scores

        self.observation_space = spaces.Dict({
            "houses": spaces.MultiDiscrete([MAX_TEMP_NUMBER + 1,] * self.SIZE),
            "fences": spaces.MultiBinary(self.NUM_FENCES),
            "pools": spaces.MultiBinary(self.NUM_POOLS),
            "parks": spaces.Discrete(self.NUM_PARKS),
        })

        '''
            For each of the three visible action/number pairs [a,n], the action mask is defined as the
            following bool ndarrays, flattened and concatenated together:
            -   A (size,) array with each True (x) element representing writing n into slot x and,
                optionally, advancing the park counter iff a is PARK or circling the pool and advancing
                the pool counter iff a is POOL and x has a pool
            -   A (size, 2, size-1) array of BIS actions with each True (x,y,z) element representing
                writing n into slot x and duplicating the house in slot z-y+1 into slot y+z
            -   A (size, 4) array of TEMP actions with each True (x,y) element representing writing
                n+y-2 iff y < 2 or n+y-1 iff y >= 2 into slot x
            -   A (size, 6) array of ESTATE actions where each True (x,y) element represents writing
                n into slot x and advancing the y-th estate score
            -   A (size, size-1) array of FENCE actions where each True (x,y) element represents
                writing n into slot x and placing a fence between the y and y+1 slots
        '''

        house_only_action = {"type": ActionCard.NONE, "shape": (self.SIZE,)}
        bis_action = {"type": ActionCard.BIS,
                      "shape": (self.SIZE, 2, self.SIZE-1)}
        estate_action = {"type": ActionCard.ESTATE, "shape": (
            self.SIZE, self.global_scores.NUM_ESTATE_TYPES)}
        temp_action = {"type": ActionCard.TEMP, "shape": (self.SIZE, 4)}
        fence_action = {"type": ActionCard.FENCE,
                        "shape": (self.SIZE, self.NUM_FENCES)}
        self.ACTIONS = [
            house_only_action,
            bis_action,
            temp_action,
            estate_action,
            fence_action
        ]
        self.ACTION_MAP = {
            ActionCard.NONE: house_only_action,
            ActionCard.PARK: house_only_action,
            ActionCard.POOL: house_only_action,
            ActionCard.BIS: bis_action,
            ActionCard.ESTATE: estate_action,
            ActionCard.FENCE: fence_action,
            ActionCard.TEMP: temp_action
        }

        offset = 0

        for action in self.ACTIONS:
            action_size = np.prod(action["shape"])
            action["slice"] = slice(offset, offset+action_size)
            action["offset"] = offset
            offset += action_size

        self.NUM_ACTIONS = offset

        neighbors = [np.eye(self.SIZE, self.SIZE-1, -1, dtype=bool),
                     np.eye(self.SIZE, self.SIZE-1, dtype=bool)]
        self.BIS_NEIGHBORS = np.stack(neighbors, axis=1)
        self.BIS_NEIGHBOR_MASK = np.invert(
            np.stack(list(reversed(neighbors)), axis=1))

    def reset(self):
        self.houses = np.zeros(self.SIZE, dtype=np.uint8)
        self.fences = np.zeros(self.NUM_FENCES, dtype=bool)
        self.pools = np.zeros(self.NUM_POOLS, dtype=bool)
        self.parks = 0
        self.valid_ranges = np.ones((self.SIZE, MAX_TEMP_NUMBER), dtype=bool)
        self.fence_mask = np.ones(self.NUM_FENCES, dtype=bool)
        self.bis_cache = None

    def get_observation(self):
        return {
            "houses": self.houses,
            "fences": self.fences,
            "pools": self.pools,
            "parks": self.parks
        }

    def get_action_mask(self, action_card: ActionCard, house_number: int) -> np.ndarray[bool]:
        house_actions = self.valid_ranges[:, house_number]

        ret = np.zeros(self.NUM_ACTIONS, dtype=bool)

        if not house_actions.any() and action_card != ActionCard.TEMP:
            return ret

        action_mask = None

        match action_card:
            case ActionCard.TEMP:
                left_bound = max(house_number-2, 0)
                if not self.valid_ranges[:, left_bound:(house_number+3)].any():
                    return ret
                action_mask = np.zeros(
                    self.ACTION_MAP[ActionCard.TEMP]["shape"], dtype=bool)
                action_mask[:, (left_bound-house_number+2):2] = self.valid_ranges[:, left_bound:house_number]
                action_mask[:, 2:] = self.valid_ranges[:,
                                                       (house_number+1):(house_number+3)]

            case ActionCard.BIS:
                action_mask = np.logical_and(house_actions[..., np.newaxis, np.newaxis], self.generate_bis_actions(
                ))
            case ActionCard.FENCE:
                action_mask = np.logical_and(house_actions[..., np.newaxis], (~self.fences & self.fence_mask)[
                    np.newaxis, ...])
            case ActionCard.ESTATE:
                action_mask = np.logical_and(
                    house_actions[..., np.newaxis], self.global_scores.get_estate_mask()[np.newaxis, ...])

        if action_mask is not None:
            ret[self.ACTION_MAP[action_card]["slice"]] = action_mask.ravel()

        ret[self.ACTION_MAP[ActionCard.NONE]["slice"]] = house_actions.ravel()
        return ret

    def generate_bis_actions(self) -> np.ndarray[bool]:

        if self.bis_cache is not None:
            return self.bis_cache
        current_house_mask = self.BIS_NEIGHBOR_MASK[self.houses.nonzero()].all(
            axis=0)
        fence_mask = np.invert(np.eye(
            self.SIZE-1, dtype=bool))[self.fences.nonzero()].all(axis=0)
        current_house_neighbors = self.BIS_NEIGHBORS[self.houses.nonzero()].any(
            axis=0)
        all_neighbors = np.logical_or(
            self.BIS_NEIGHBORS, current_house_neighbors[np.newaxis, ...])
        self.bis_cache = np.logical_and.reduce(np.broadcast_arrays(
            all_neighbors, current_house_mask[np.newaxis, ...], fence_mask[np.newaxis, np.newaxis, ...], self.BIS_NEIGHBOR_MASK))
        return self.bis_cache

    def apply_action(self, action_type: ActionCard, house_number: int, encoded_action: int) -> None:
        decoded_action_type, (slot, *rest) = self.decode_action(encoded_action)

        if action_type == decoded_action_type:
            match action_type:
                case ActionCard.ESTATE:
                    self.global_scores.advance_estate(rest[0])
                case ActionCard.FENCE:
                    assert self.fence_mask[rest[0]]
                    self.fences[rest[0]] = True
                case ActionCard.BIS:
                    self.add_house(slot, house_number)
                    source_slot = rest[1] - rest[0] + 1
                    destination_slot = rest[0] + rest[1]
                    assert self.houses[source_slot] > 0

                    self.add_house(destination_slot,
                                   self.houses[source_slot]-1, bis=True)
                    self.global_scores.advance_bis()

                    assert not self.fences[rest[1]]
                    self.fence_mask[rest[1]] = False
                case ActionCard.TEMP:
                    house_number += Row.TEMP_OFFSET[rest[0]]
                case _:
                    assert False
        else:
            assert decoded_action_type == ActionCard.NONE

            match action_type:
                case ActionCard.POOL if slot in self.POOL_IDX:
                    idx = self.POOL_IDX.index(slot)
                    if not self.pools[idx]:
                        self.pools[idx] = True
                        self.global_scores.advance_pool()
                case ActionCard.PARK if self.parks < (self.NUM_PARKS - 1):
                    self.parks += 1

        if decoded_action_type != ActionCard.BIS:
            self.add_house(slot, house_number)

        self.bis_cache = None

    def decode_action(self, action: int) -> tuple[ActionCard, tuple[int, ...]]:
        for action_desc in reversed(self.ACTIONS):
            if action >= action_desc["offset"]:
                return action_desc["type"], np.unravel_index(action - action_desc["offset"], action_desc["shape"])

    def add_house(self, slot: int, house_number: int, bis=False) -> None:
        assert self.houses[slot] == 0
        assert bis or self.valid_ranges[slot, house_number]

        self.houses[slot] = house_number + 1
        self.valid_ranges[:slot+1, house_number:] = False
        self.valid_ranges[slot:, :house_number+1] = False

    def get_score(self) -> int:
        return (self.get_estates() * self.global_scores.get_estate_scores()).sum() + self.PARK_SCORES[self.parks]

    def get_estates(self) -> np.ndarray[int]:
        estates = np.split(self.houses > 0, self.fences.nonzero()[0] + 1)
        ret = np.zeros(self.global_scores.NUM_ESTATE_TYPES, dtype=np.uint8)
        for e in estates:
            if not e.all() or len(e) > self.global_scores.NUM_ESTATE_TYPES:
                continue
            ret[len(e) - 1] += 1
        return ret

    def is_full(self) -> bool:
        return np.all(self.houses > 0)


class Deck:

    def __init__(self, cards: dict):
        d = []
        for (a, cl) in cards.items():
            for c in cl:
                d.append((a, c))
        self.deck = np.array(d, dtype=np.uint8)
        self.NUM_CARDS = len(self.deck)

        self.observation_space = {
            "visible_actions": spaces.MultiDiscrete([len(ActionCard) - 1] * 3),
            "visible_actions": spaces.MultiDiscrete([MAX_HOUSE_NUMBER] * 3),
            "visible_next_actions": spaces.MultiDiscrete([len(ActionCard) - 1] * 3),
            "remaining_triplets": spaces.Discrete((self.NUM_CARDS // 3) - 1)
        }

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.rng.shuffle(self.deck)
        self.deck_pointer = 3

    def get_observation(self):
        action_cards = self.deck[(self.deck_pointer - 3):self.deck_pointer]
        number_cards = self.deck[self.deck_pointer: (self.deck_pointer+3)]
        return {
            "visible_actions": action_cards[:, 0],
            "visible_numbers": number_cards[:, 1],
            "visible_next_actions": number_cards[:, 0],
            "remaining_triplets": ((self.NUM_CARDS - self.deck_pointer) // 3) - 1
        }

    def get_visible_cards(self):
        actions = self.deck[(self.deck_pointer - 3):self.deck_pointer, 0]
        numbers = self.deck[self.deck_pointer: (self.deck_pointer+3), 1]
        return np.column_stack((actions, numbers))

    def advance_cards(self):
        if self.deck_pointer == (self.NUM_CARDS - 3):
            # swap first and last triple
            self.deck[:3], self.deck[-3:] = self.deck[-3:], self.deck[:3].copy()
            self.rng.shuffle(self.deck[3:])
            self.deck_pointer = 3
        else:
            self.deck_pointer += 3


DEFAULT_CONFIG = {
    "deck": {
        ActionCard.POOL: (2, 3, 5, 6, 7, 8, 9, 11, 12),
        ActionCard.TEMP: (2, 3, 5, 6, 7, 8, 9, 11, 12),
        ActionCard.BIS: (2, 3, 5, 6, 7, 8, 9, 11, 12),
        ActionCard.PARK: (0, 1, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 13, 14),
        ActionCard.ESTATE: (0, 1, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 13, 14),
        ActionCard.FENCE: (0, 1, 2, 4, 4, 5,
                           5, 6, 7, 7, 8, 9, 9, 10, 10, 12, 13, 14)
    },
    "rows": (
        {"size": 10, "parks": (0, 2, 4, 6, 10), "pools": (2, 6, 7)},
        {"size": 11, "parks": (0, 2, 4, 6, 8, 14), "pools": (0, 3, 7)},
        {"size": 12, "parks": (0, 2, 4, 6, 8, 10, 18), "pools": (1, 6, 10)}
    ),
    "global_scores": {
        "estate_scores": ((1, 3), (2, 3, 4), (3, 4, 5, 6), (
            4, 5, 6, 7, 8), (5, 6, 7, 8, 10), (6, 7, 8, 10, 12)
        ),
        "bis_scores": (0, -1, -3, -6, -9, -12, -16, -20, -24, -28),
        "pool_scores": (0, 3, 6, 9, 13, 17, 21, 26, 31, 36),
        "failure_scores": (0, 0, -3, -5)
    }
}


class WelcomeToEnv(Env):

    def __init__(self, config=DEFAULT_CONFIG):
        self.global_scores = GlobalScores(**config["global_scores"])
        self.rows = tuple(Row(**kwargs, global_scores=self.global_scores)
                          for kwargs in config["rows"])

        self.rng = np_random()
        self.deck = Deck(config["deck"])

        self.hidden_observation_space = spaces.Dict({
            "rows": spaces.Tuple([r.observation_space for r in self.rows]),
        } | self.global_scores.observation_space | self.deck.observation_space)
        self.observation_space = spaces.flatten_space(
            self.hidden_observation_space)

        self.ROW_BINS = np.cumsum([r.NUM_ACTIONS for r in self.rows])

        self.NUM_ROW_ACTIONS = self.ROW_BINS[-1]

        # additional null action, when all the others are impossible
        self.NUM_ACTIONS = (self.NUM_ROW_ACTIONS * 3) + 1

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        self.action_space.sample = self.sample_action

    def sample_action(self):
        return self.rng.choice(self.action_masks().nonzero()[0])

    def seed(self, seed):
        self.rng = np_random(seed)

    def reset(self):
        self.global_scores.reset()
        for r in self.rows:
            r.reset()
        self.deck.reset(self.rng)
        self.cumulative_score = 0.
        return self.get_observation()

    def get_observation(self) -> dict:
        return spaces.flatten(self.hidden_observation_space, {
            "rows": tuple(r.get_observation() for r in self.rows),
        } | self.global_scores.get_observation() | self.deck.get_observation())

    def get_score(self) -> int:
        return sum((r.get_score() for r in self.rows)) + self.global_scores.get_global_score()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        assert self.action_masks()[action]
        done = False
        if action == self.NUM_ACTIONS - 1:
            self.global_scores.advance_failure_score()
            done = self.global_scores.is_max_failure_reached()
        else:
            card_idx, row_offset = np.unravel_index(
                action, (3, self.NUM_ROW_ACTIONS))
            card = self.deck.get_visible_cards()[card_idx]
            row_idx = np.digitize(row_offset, self.ROW_BINS)
            row = self.rows[row_idx]
            row.apply_action(*card, row_offset -
                             self.ROW_BINS[row_idx] + row.NUM_ACTIONS)

        done = done or all([r.is_full() for r in self.rows])
        reward = float(self.get_score() - self.cumulative_score)
        self.cumulative_score += reward

        self.deck.advance_cards()

        return self.get_observation(), reward, done, {}

    def action_masks(self) -> np.ndarray[bool]:
        visible_cards = self.deck.get_visible_cards()
        ret = np.zeros(self.NUM_ACTIONS, dtype=bool)
        row_offset = 0
        for row in self.rows:
            for i, (action, number) in enumerate(visible_cards):
                offset = i * self.NUM_ROW_ACTIONS + row_offset
                ret[offset:(offset+row.NUM_ACTIONS)
                    ] = row.get_action_mask(action, number)
            row_offset += row.NUM_ACTIONS
        ret[-1] = not ret.any()
        return ret


def np_random(seed: Optional[int] = None) -> np.random.Generator:
    """Generates a random number generator from the seed and returns the Generator and seed.
    Args:
        seed: The seed used to create the generator
    Returns:
        The generator and resulting seed
    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    seed_seq = np.random.SeedSequence(seed)
    return np.random.Generator(np.random.PCG64(seed_seq))
