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


ESTATE_SCORES = [[1, 3], [2, 3, 4], [3, 4, 5, 6], [
    4, 5, 6, 7, 8], [5, 6, 7, 8, 10], [6, 7, 8, 10, 12]]
POOL_LOCATIONS = [[3, 7, 8], [1, 4, 8], [2, 7, 11]]
PARK_SCORES = [[0, 2, 4, 6, 10], [0, 2, 4, 6, 8, 14], [0, 2, 4, 6, 8, 10, 18]]
POOL_SCORING = [0, 3, 6, 9, 13, 17, 21, 26, 31, 36]


def pad_2d_array(arr: Iterable[Sized]) -> np.ndarray:
    max_len = max([len(x) for x in arr])
    return np.stack([np.pad(np.array(x), (0, max_len - len(x))) for x in arr])


class GlobalScores:
    BIS_SCORING = [0, -1, -3, -6, -9, -12, -16, -20, -24, -28]
    POOL_SCORING = [0, 3, 6, 9, 13, 17, 21, 26, 31, 36]
    ESTATE_SCORES = [[1, 3], [2, 3, 4], [3, 4, 5, 6], [
        4, 5, 6, 7, 8], [5, 6, 7, 8, 10], [6, 7, 8, 10, 12]]
    FAILURE_TO_PLAY_SCORES = [0, 0, -3, -5]

    NUM_ESTATE_TYPES = len(ESTATE_SCORES)

    MAX_ESTATE_COUNT = np.array([len(x)-1
                                for x in ESTATE_SCORES], dtype=np.uint8)

    MAX_ESTATE_SCORE_LEN = max([len(x) for x in ESTATE_SCORES])

    ESTATE_SCORES_ARRAY = pad_2d_array(ESTATE_SCORES)

    observation_space = {
        "bis_score": spaces.Discrete(len(BIS_SCORING)),
        "pool_score": spaces.Discrete(len(POOL_SCORING)),
        "estate_scores": spaces.MultiDiscrete([len(x) for x in ESTATE_SCORES]),
        "failure_to_play": spaces.Discrete(len(FAILURE_TO_PLAY_SCORES))
    }

    def reset(self):
        self.pool_score = 0
        self.bis_score = 0
        self.estate_scores = np.zeros(
            GlobalScores.NUM_ESTATE_TYPES, dtype=np.uint8)
        self.failure_to_play = 0

    def get_observation(self):
        return {
            "bis_score": self.bis_score,
            "pool_score": self.pool_score,
            "estate_scores": self.estate_scores,
            "failure_to_play": self.failure_to_play
        }

    def advance_pool(self):
        if self.pool_score < len(GlobalScores.POOL_SCORING) - 1:
            self.pool_score += 1

    def advance_bis(self):
        if self.bis_score < len(GlobalScores.BIS_SCORING) - 1:
            self.bis_score += 1

    def advance_estate(self, estate_num: int):
        self.estate_scores[estate_num] += 1

    def advance_failure_to_play(self):
        self.failure_to_play += 1

    def is_max_failure_reached(self):
        return self.failure_to_play == len(GlobalScores.FAILURE_TO_PLAY_SCORES) - 1

    def get_estate_mask(self) -> np.ndarray[bool]:
        return self.estate_scores < GlobalScores.MAX_ESTATE_COUNT

    def get_pool_mask(self) -> bool:
        return self.pool_score < len(GlobalScores.POOL_SCORING) - 1

    def get_estate_scores(self) -> list[int]:
        return GlobalScores.ESTATE_SCORES_ARRAY[np.mgrid[:GlobalScores.NUM_ESTATE_TYPES], self.estate_scores]

    def get_pool_score(self) -> int:
        return GlobalScores.POOL_SCORING[self.pool_score]

    def get_bis_score(self) -> int:
        return GlobalScores.BIS_SCORING[self.bis_score]

    def get_failure_to_play_score(self) -> int:
        return GlobalScores.FAILURE_TO_PLAY_SCORES[self.failure_to_play]

    def get_global_score(self) -> int:
        return self.get_pool_score() + self.get_bis_score() + self.get_failure_to_play_score()


class Row:
    HOUSE_NUMBER_DIAG = np.pad(
        np.eye(MAX_TEMP_NUMBER, dtype=bool), ((1, 0), (0, 0)))

    def __init__(self, /, size: int, pools: Iterable[int], park_scores: Iterable[int], global_scores: GlobalScores):
        self.SIZE = size
        self.POOL_IDX = tuple(pools)
        self.NUM_PARKS = len(park_scores)
        self.NUM_FENCES = self.SIZE - 1
        self.NUM_POOLS = len(self.POOL_IDX)
        self.PARK_SCORES = tuple(park_scores)

        self.global_scores = global_scores

        self.observation_space = spaces.Dict({
            "houses": spaces.MultiDiscrete([MAX_TEMP_NUMBER + 1,] * self.SIZE),
            "fences": spaces.MultiBinary(self.NUM_FENCES),
            "pools": spaces.MultiBinary(self.NUM_POOLS),
            "parks": spaces.Discrete(self.NUM_PARKS),
        })

        '''
            Internally, the action mask is represented as the following bool ndarrays, flattened
            and concatenated together:
            -   A (size, 17) array of temp actions with each True (i,j) element representing writing
                house number j+1 in slot i and advancing the temp counter
            -   A (size, 15, 2, size-1) array of BIS actions, with each True (i,j,x,y) element representing
                writing house number j+1 in slot i, duplicating the house in slot y-x+1 into slot y+x,
                and advancing the BIS counter
            -   A (3, 15) array of pool actions with each True (i,j) element representing writing house
                number j+1 in the i-th pool slot, in addition to circling the pool and advancing the
                pool counter
            -   A (size, 15, size-1) array of fence actions, with each True (i,j,k) element representing
                writing house number j+1 in slot i and placing a fence between the k and k+1 spots
            -   A (size, 15) array of park actions with each True (i,j) element representing writing
                house number j+1 in slot i and advancing the park counter
            -   A (size, 15, 6) array of estate actions, with each True (i,j,k) element representing
                writing house number j+1 in slot i and advancing the k+1-th estate score
            -   A (size, 15) array of house only actions with each True (i,j) element representing
                writing house number j+1 in slot i and nothing else
        '''

        self.ACTIONS = [None] * len(ActionCard)
        self.ACTIONS[ActionCard.TEMP] = {
            "shape": (self.SIZE, MAX_TEMP_NUMBER)}
        self.ACTIONS[ActionCard.BIS] = {
            "shape": (self.SIZE, MAX_HOUSE_NUMBER, 2, self.SIZE - 1)}
        self.ACTIONS[ActionCard.POOL] = {
            "shape": (self.NUM_POOLS, MAX_HOUSE_NUMBER)}
        self.ACTIONS[ActionCard.FENCE] = {
            "shape": (self.SIZE, MAX_HOUSE_NUMBER, self.NUM_FENCES)}
        self.ACTIONS[ActionCard.PARK] = {
            "shape": (self.SIZE, MAX_HOUSE_NUMBER)}
        self.ACTIONS[ActionCard.ESTATE] = {"shape": (
            self.SIZE, MAX_HOUSE_NUMBER, GlobalScores.NUM_ESTATE_TYPES)}
        self.ACTIONS[ActionCard.NONE] = {
            "shape": (self.SIZE, MAX_HOUSE_NUMBER)}

        offset = 0

        for action in self.ACTIONS:
            action_size = np.prod(action["shape"])
            action["slice"] = slice(offset, offset+action_size)
            action["offset"] = offset
            offset += action_size

        self.NUM_ACTIONS = offset

        self.null_house_action = np.zeros(
            self.ACTIONS[ActionCard.NONE]["shape"], dtype=bool)

        self.null_pool_action = np.zeros(
            self.ACTIONS[ActionCard.POOL]["shape"], dtype=bool)

        neighbors = [np.eye(self.SIZE, self.SIZE-1, -1, dtype=bool),
                     np.eye(self.SIZE, self.SIZE-1, dtype=bool)]
        self.BIS_NEIGHBORS = np.stack(neighbors, axis=1)
        self.BIS_NEIGHBOR_MASK = np.invert(
            np.stack(reversed(neighbors), axis=1))

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
        new_house_num = np.eye(MAX_HOUSE_NUMBER, dtype=bool)[house_number]
        house_actions = np.logical_and(
            self.valid_ranges[:, :MAX_HOUSE_NUMBER], new_house_num[np.newaxis, ...])

        ret = np.zeros(self.NUM_ACTIONS, dtype=bool)
        if not house_actions.any():
            return ret

        match action_card:
            case ActionCard.BIS:
                action_mask = np.where(house_actions[..., np.newaxis, np.newaxis], self.generate_bis_actions(
                )[:, np.newaxis, ...], False)
            case ActionCard.FENCE:
                action_mask = np.where(house_actions[..., np.newaxis], (~self.fences & self.fence_mask)[
                                       np.newaxis, np.newaxis, ...], False)
            case ActionCard.TEMP:
                new_house_nums = np.eye(MAX_TEMP_NUMBER, dtype=bool)[
                    house_number]
                new_house_nums[(max(house_number - 2, 0))                               :(min(house_number + 3, MAX_HOUSE_NUMBER))] = True
                action_mask = np.logical_and(self.valid_ranges, new_house_nums)
            case ActionCard.POOL:
                action_mask = house_actions[self.POOL_IDX, :] if self.global_scores.get_pool_mask(
                ) else self.null_pool_action
            case ActionCard.ESTATE:
                action_mask = np.where(
                    house_actions[..., np.newaxis], self.global_scores.get_estate_mask()[np.newaxis, np.newaxis, ...], False)
            case ActionCard.PARK:
                action_mask = house_actions if self.parks < (
                    self.NUM_PARKS-1) else self.null_house_action

        ret[self.ACTIONS[action_card]["slice"]] = action_mask.ravel()
        ret[self.ACTIONS[ActionCard.NONE]["slice"]] = house_actions.ravel()
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

    def apply_action(self, action: int) -> None:
        action_type, (slot, house_number, *rest) = self.decode_action(action)
        match action_type:
            case ActionCard.POOL:
                self.pools[slot] = True
                slot = self.POOL_IDX[slot]
                self.global_scores.advance_pool()
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
            case ActionCard.PARK:
                self.parks += 1

        if action_type != ActionCard.BIS:
            self.add_house(slot, house_number)

        self.bis_cache = None

    def decode_action(self, action: int) -> tuple[ActionCard, tuple[int, ...]]:
        for action_type in reversed(ActionCard):
            action_desc = self.ACTIONS[action_type]
            if action >= action_desc["offset"]:
                return action_type, np.unravel_index(action - action_desc["offset"], action_desc["shape"])

    def add_house(self, slot: int, house_number: int, bis=False) -> None:
        assert self.houses[slot] == 0
        assert bis or self.valid_ranges[slot, house_number]

        self.houses[slot] = house_number + 1
        new_mask = np.ones_like(self.valid_ranges, dtype=bool)
        new_mask[:slot+1, house_number:] = False
        new_mask[slot:, :house_number+1] = False
        self.valid_ranges &= new_mask

    def get_park_score(self):
        return self.PARK_SCORES[self.parks]

    def get_score(self) -> int:
        return (self.get_estates() * self.global_scores.get_estate_scores()).sum() + self.PARK_SCORES[self.parks]

    def get_estates(self) -> np.ndarray[int]:
        estates = np.split(self.houses > 0, self.fences.nonzero()[0] + 1)
        ret = np.zeros(GlobalScores.NUM_ESTATE_TYPES, dtype=np.uint8)
        for e in estates:
            if not e.all() or len(e) > GlobalScores.NUM_ESTATE_TYPES:
                continue
            ret[len(e) - 1] += 1
        return ret

    def is_full(self) -> bool:
        return np.all(self.houses > 0)


class Deck:

    def create_deck_mapping():
        # creates a (6,18) array that maps linear card numbers to specific house
        # numbers for each action type
        mapping = [None] * (len(ActionCard) - 1)
        mapping[ActionCard.POOL] = mapping[ActionCard.TEMP] = mapping[ActionCard.BIS] = [
            2, 3, 5, 6, 7, 8, 9, 11, 12]
        mapping[ActionCard.PARK] = mapping[ActionCard.ESTATE] = [
            0, 1, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 10, 11, 13, 14]
        mapping[ActionCard.FENCE] = [0, 1, 2, 4, 4, 5,
                                     5, 6, 7, 7, 8, 9, 9, 10, 10, 12, 13, 14]

        # creates an (81,2) array, where each element [n,m] represents a card of action
        # type n and house number DECK_MAP[n][m]
        deck = np.concatenate([np.mgrid[i:i+1, :len(x)].T.reshape(-1, 2)
                               for i, x in enumerate(mapping)])

        return deck, pad_2d_array(mapping)

    NEW_DECK, DECK_MAP = create_deck_mapping()
    NUM_CARDS = len(NEW_DECK)

    observation_space = {
        "visible_cards": spaces.MultiBinary((len(ActionCard) - 1, MAX_HOUSE_NUMBER)),
        "visible_next_actions": spaces.MultiDiscrete([len(ActionCard) - 1] * 3),
        "remaining_triplets": spaces.Discrete((NUM_CARDS // 3) - 1)
    }

    def __init__(self):
        self.deck = Deck.NEW_DECK.copy()

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.rng.shuffle(self.deck)
        self.deck_pointer = 3

    def get_observation(self):
        action_cards = self.deck[(self.deck_pointer - 3):self.deck_pointer]
        number_cards = self.deck[self.deck_pointer: (self.deck_pointer+3)]
        number_card_actions = number_cards[:, 0]
        visible_numbers = Deck.DECK_MAP[number_card_actions][[
            0, 1, 2], number_cards[:, 1]],
        visible_cards = np.zeros(
            (len(ActionCard) - 1, MAX_HOUSE_NUMBER), dtype=bool)
        visible_cards[action_cards[:, 0], visible_numbers] = True
        return {
            # "visible_actions": action_cards[:, 0],
            # "visible_numbers": Deck.DECK_MAP[number_card_actions][[0, 1, 2], number_cards[:, 1]],
            "visible_cards": visible_cards,
            "visible_next_actions": number_card_actions,
            "remaining_triplets": ((Deck.NUM_CARDS - self.deck_pointer) // 3) - 1
        }

    def advance_cards(self):
        if self.deck_pointer == (Deck.NUM_CARDS - 3):
            self.deck = np.concatenate((self.deck[-3:], self.deck[:-3]))
            self.rng.shuffle(self.deck[3:])
            self.deck_pointer = 3
        else:
            self.deck_pointer += 3


class WelcomeToEnv(Env):

    def __init__(self):
        self.global_scores = GlobalScores()
        # self.rows = (Row(size=10, pools=[
        #              2, 6, 7], park_scores=PARK_SCORES[0], global_scores=self.global_scores),
        #              Row(size=11, pools=[
        #                  0, 3, 7], park_scores=PARK_SCORES[1], global_scores=self.global_scores),
        #              Row(size=12, pools=[1, 6, 10], park_scores=PARK_SCORES[2], global_scores=self.global_scores))
        self.rows = (Row(size=3, pools=[1], park_scores=[
                     0, 2, 4], global_scores=self.global_scores),)

        self.rng = np_random()
        self.deck = Deck()

        self.hidden_observation_space = spaces.Dict({
            "rows": spaces.Tuple([r.observation_space for r in self.rows]),
        } | self.global_scores.observation_space | self.deck.observation_space)
        self.observation_space = spaces.flatten_space(
            self.hidden_observation_space)

        offset = 0
        self.action_offsets = []
        for r in self.rows:
            self.action_offsets.append(offset)
            offset += r.NUM_ACTIONS

        # additional null action, when all the others are impossible
        self.NUM_ACTIONS = offset + 1

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
            self.global_scores.advance_failure_to_play()
            done = self.global_scores.is_max_failure_reached()
        else:
            for i, offset in reversed(list(enumerate(self.action_offsets))):
                if action >= offset:
                    self.rows[i].apply_action(action - offset)
                    break

        done = done or all([r.is_full() for r in self.rows])
        reward = float(self.get_score() - self.cumulative_score)
        self.cumulative_score += reward

        self.deck.advance_cards()

        return self.get_observation(), reward, done, {}

    def action_masks(self) -> np.ndarray[bool]:
        visible_cards = np.column_stack(self.deck.get_observation()[
                                        "visible_cards"].nonzero())
        ret = np.zeros(self.NUM_ACTIONS, dtype=bool)
        offset = 0
        for row in self.rows:
            ret[offset:(offset+row.NUM_ACTIONS)] = np.logical_or.reduce([row.get_action_mask(
                action, number) for action, number in visible_cards])
            offset += row.NUM_ACTIONS
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


gs = GlobalScores()
gs.reset()
r = Row(size=10, pools=[2, 6, 7],
        park_scores=PARK_SCORES[0], global_scores=gs)
r.reset()
r.add_house(1, 1)
r.add_house(2, 2)
# r.fences[[1, 4]] = True
r.get_action_mask(action_card=ActionCard.BIS, house_number=5)
env = WelcomeToEnv()
env.reset()
env.action_masks()
