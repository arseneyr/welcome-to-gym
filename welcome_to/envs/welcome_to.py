from collections import namedtuple
from enum import IntEnum
from typing import Iterable, Optional, Tuple
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

    # ESTATE_SCORES_ARRAY = np.stack([np.pad(np.array(
    #     x), (0, MAX_ESTATE_SCORE_LEN - len(x)), constant_values=x[-1]) for x in ESTATE_SCORES])

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
        return GlobalScores.ESTATE_SCORES_ARRAY[self.estate_scores]

    def get_pool_score(self) -> int:
        return GlobalScores.POOL_SCORING[self.pool_score]

    def get_bis_score(self) -> int:
        return GlobalScores.BIS_SCORING[self.bis_score]

    def get_failure_to_play_score(self) -> int:
        return GlobalScores.FAILURE_TO_PLAY_SCORES[self.failure_to_play]

    def get_global_score(self) -> int:
        return self.get_pool_score() + self.get_bis_score() + self.get_failure_to_play_score


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
            "houses": spaces.MultiDiscrete([MAX_HOUSE_NUMBER + 1,] * self.SIZE),
            "fences": spaces.MultiBinary(self.NUM_FENCES),
            "pools": spaces.MultiBinary(self.NUM_POOLS),
            "parks": spaces.Discrete(self.NUM_PARKS),
        })

        '''
            Internally, the action space is represented as the following ndarrays, flattened
            and concatenated together:
            -   A (size, 17) array of temp actions with each (i,j) element representing writing
                house number j+1 in slot i and advancing the temp counter
            -   A (size, 15, size) array of BIS actions, with each (i,j,k) element representing
                writing house number j+1 in slot i, duplicating the lowest possible numbered house
                into slot k, and advancing the BIS counter
            -   A (3, 15) array of pool actions with each (i,j) element representing writing house
                number j+1 in the i-th pool slot, in addition to circling the pool and advancing the
                pool counter
            -   A (size, 15, size-1) array of fence actions, with each (i,j,k) element representing
                writing house number j+1 in slot i and placing a fence between the k and k+1 spots
            -   A (size, 15) array of park actions with each (i,j) element representing writing
                house number j+1 in slot i and advancing the park counter
            -   A (size, 15, 6) array of estate actions, with each (i,j,k) element representing
                writing house number j+1 in slot i and advancing the k+1-th estate score
            -   A (size, 15) array of house only actions with each (i,j) element representing
                writing house number j+1 in slot i and nothing else
        '''

        self.ACTIONS = [None] * len(ActionCard)
        self.ACTIONS[ActionCard.TEMP] = {
            "shape": (self.SIZE, MAX_TEMP_NUMBER)}
        self.ACTIONS[ActionCard.BIS] = {
            "shape": (self.SIZE, MAX_HOUSE_NUMBER, self.SIZE)}
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

        self.bis_neighbors = np.zeros(
            (self.SIZE, MAX_HOUSE_NUMBER, self.SIZE, MAX_TEMP_NUMBER), dtype=bool)
        self.NEIGHBOR_DIAGS = np.eye(
            self.SIZE, self.SIZE, 1, dtype=bool) + np.eye(self.SIZE, self.SIZE, -1, dtype=bool)
        idx = np.ogrid[:self.SIZE, :MAX_HOUSE_NUMBER]
        self.bis_neighbors[idx[0], idx[1], :,
                           idx[1]] = self.NEIGHBOR_DIAGS[idx[0]]
        self.bis_neighbor_mask = np.ones(
            (self.SIZE, 1, self.SIZE, MAX_TEMP_NUMBER), dtype=bool)
        self.bis_neighbor_mask[idx[0], 0, idx[0], :] = False

    def reset(self):
        self.houses = np.zeros(self.SIZE, dtype=np.uint8)
        self.fences = np.zeros(self.NUM_FENCES, dtype=bool)
        self.pools = np.zeros(self.NUM_POOLS, dtype=bool)
        self.parks = 0
        self.valid_ranges = np.ones((self.SIZE, MAX_TEMP_NUMBER), dtype=bool)
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
                action_mask = np.where(house_actions[..., np.newaxis], self.generate_bis_actions(
                ).max(axis=3), False)
            case ActionCard.FENCE:
                action_mask = np.where(house_actions[..., np.newaxis], np.invert(
                    self.fences)[np.newaxis, np.newaxis, ...], False)
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
                action_mask = house_actions if self.parks < self.NUM_PARKS else self.null_house_action

        ret[self.ACTIONS[action_card]["slice"]] = action_mask.ravel()
        ret[self.ACTIONS[ActionCard.NONE]["slice"]] = house_actions.ravel()
        return ret

    def generate_bis_actions(self) -> np.ndarray[bool]:
        if self.bis_cache is not None:
            return self.bis_cache
        empty_slots = self.houses == 0
        house_mask = np.logical_and(self.bis_neighbor_mask, empty_slots[
            np.newaxis, np.newaxis, :, np.newaxis])
        expanded_house_map = Row.HOUSE_NUMBER_DIAG[self.houses]
        neighbors_xor = np.pad(np.logical_xor(
            expanded_house_map[1:, :], expanded_house_map[:-1, :]), ((1, 1), (0, 0)))
        current_neighbors = np.logical_or(
            neighbors_xor[1:, :], neighbors_xor[:-1, :])
        self.bis_cache = np.logical_and(np.logical_or(
            self.bis_neighbors, current_neighbors[np.newaxis, np.newaxis, ...]), house_mask)
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
                self.fences[rest[0]] = True
            case ActionCard.BIS:
                bis_slot = rest[0]
                bis_house_number = np.argmax(self.generate_bis_actions()[
                                             slot, house_number, bis_slot])
                self.add_house(bis_slot, bis_house_number)
                self.bis_controller.advance_bis()
            case ActionCard.PARK:
                self.parks += 1

        self.add_house(slot, house_number)
        self.bis_cache = None

    def decode_action(self, action: int) -> tuple[ActionCard, tuple[int, ...]]:
        for action_type in reversed(ActionCard):
            action_desc = self.ACTIONS[action_type]
            if action >= action_desc["offset"]:
                return action_type, np.unravel_index(action - action_desc["offset"], action_desc["shape"])

    def add_house(self, slot: int, house_number: int) -> None:
        self.houses[slot] = house_number + 1
        new_mask = np.ones_like(self.valid_ranges, dtype=bool)
        new_mask[:slot+1, house_number:] = False
        new_mask[slot:, :house_number+1] = False
        self.valid_ranges &= new_mask

    def get_park_score(self):
        return self.PARK_SCORES[self.parks]

    def get_score(self) -> int:
        return (self.get_estates() * self.global_scores.estate_scores()).sum() + self.PARK_SCORES[self.parks]

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

        max_len = max([len(x) for x in mapping])

        # creates a (6,18) array that maps linear card numbers to specific house
        # numbers for each action type
        mapping = np.stack([np.pad(x, (0, max_len - len(x))) for x in mapping])
        return deck, mapping

    NEW_DECK, DECK_MAP = create_deck_mapping()
    NUM_CARDS = len(NEW_DECK)

    observation_space = {
        "visible_actions": spaces.MultiDiscrete([len(ActionCard) - 1] * 3),
        "visible_numbers": spaces.MultiDiscrete([MAX_HOUSE_NUMBER] * 3),
        "visible_next_actions": spaces.MultiDiscrete([len(ActionCard) - 1] * 3),
        "remaining_triplets": spaces.Discrete((NUM_CARDS // 3) - 2)
    }

    def __init__(self, rng: np.random.Generator):
        self.deck = Deck.NEW_DECK.copy()
        self.rng = rng

    def reset(self):
        self.rng.shuffle(self.deck)
        self.deck_pointer = 3

    def get_observation(self):
        action_cards = self.deck[(self.deck_pointer - 3):self.deck_pointer]
        number_cards = self.deck[self.deck_pointer: (self.deck_pointer+3)]
        number_card_actions = number_cards[:, 0]
        return {
            "visible_actions": action_cards[:, 0],
            "visible_numbers": Deck.DECK_MAP[number_card_actions][[0, 1, 2], number_cards[:, 1]],
            "visible_next_actions": number_card_actions,
            "remaining_triplets": ((Deck.NUM_CARDS - self.deck_pointer - 3) / 3)
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
        self.rows = (Row(size=10, pools=[
                     2, 6, 7], park_scores=PARK_SCORES[0], global_scores=self.global_scores),
                     Row(size=11, pools=[
                         0, 3, 7], park_scores=PARK_SCORES[1], global_scores=self.global_scores),
                     Row(size=12, pools=[1, 6, 10], park_scores=PARK_SCORES[2], global_scores=self.global_scores))

        self.deck = Deck(self.np_random)

        self.observation_space = spaces.Dict({
            "rows": spaces.Tuple([r.observation_space for r in self.rows]),
        } | self.global_scores.observation_space | self.deck.observation_space)

        offset = 0
        self.action_offsets = []
        for r in self.rows:
            self.action_offsets.append(offset)
            offset += r.NUM_ACTIONS

        # additional null action, when all the others are impossible
        self.NUM_ACTIONS = offset + 1

        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.global_scores.reset()
        for r in self.rows:
            r.reset()
        self.deck.reset()
        self.cumulative_score = 0
        return self.get_observation()

    def get_observation(self) -> dict:
        return {
            "rows": (r.get_observation() for r in self.rows),
        } | self.global_scores.get_observation() | self.deck.get_observation()

    def get_score(self) -> int:
        return sum((r.get_score() for r in self.rows)) + self.global_scores.get_global_score()

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
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
        reward = self.get_score() - self.cumulative_score
        self.cumulative_score += reward

        self.deck.advance_cards()

        return self.get_observation(), reward, done, False

    def action_masks(self) -> np.ndarray[bool]:
        deck_observation = self.deck.get_observation()
        visible_cards = np.column_stack(
            (deck_observation["visible_actions"], deck_observation["visible_numbers"]))
        ret = np.zeros(self.NUM_ACTIONS, dtype=bool)
        offset = 0
        for row in self.rows:
            ret[offset:(offset+row.NUM_ACTIONS)] = np.logical_or.reduce([row.get_action_mask(
                action, number) for action, number in visible_cards])
            offset += row.NUM_ACTIONS
        ret[-1] = not ret.any()
        return ret
