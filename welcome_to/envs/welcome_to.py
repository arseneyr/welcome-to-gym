from enum import IntEnum
from typing import Iterable, Optional
import numpy as np
from gym import (
    spaces
)

MAX_HOUSE_NUMBER = 15
MAX_TEMP_NUMBER = MAX_HOUSE_NUMBER + 2


class ActionCard(IntEnum):
    NONE = 0
    TEMP = 1
    BIS = 2
    POOL = 3
    FENCE = 4
    PARK = 5
    ESTATE = 6


ESTATE_SCORES = [[1, 3], [2, 3, 4], [3, 4, 5, 6], [
    4, 5, 6, 7, 8], [5, 6, 7, 8, 10], [6, 7, 8, 10, 12]]
POOL_LOCATIONS = [[3, 7, 8], [1, 4, 8], [2, 7, 11]]
PARK_SCORES = [[0, 2, 4, 6, 10], [0, 2, 4, 6, 8, 14], [0, 2, 4, 6, 8, 10, 18]]
POOL_SCORING = [0, 3, 6, 9, 13, 17, 21, 26, 31, 36]


class Row:
    null_estate_actions = np.zeros(len(ESTATE_SCORES), dtype=bool)
    house_number_diag = np.eye(MAX_HOUSE_NUMBER, dtype=bool)
    house_number_diag_one = np.vstack(
        (np.zeros(MAX_HOUSE_NUMBER, dtype=bool), house_number_diag))

    def __init__(self, size: int, pools: Iterable[int], num_parks: int, estate_actions_view: np.ndarray[bool]):
        self.SIZE = size
        self.POOL_IDX = tuple(pools)
        self.NUM_PARKS = num_parks
        self.NUM_FENCES = size - 1
        self.NUM_POOLS = len(self.POOL_IDX)

        self.estate_actions = estate_actions_view

        self.observation_space = spaces.Dict({
            "houses": spaces.MultiDiscrete([MAX_HOUSE_NUMBER + 1,] * size),
            "fences": spaces.MultiBinary(self.NUM_FENCES),
            "pools": spaces.MultiBinary(self.NUM_POOLS),
            "parks": spaces.Discrete(self.NUM_PARKS),
        })

        '''
            Internally, the action space is represented as the following ndarrays, flattened
            and concatenated together:
            -   A (size, 15, size) array of BIS actions, with each (i,j,k) element representing
                writing house number j+1 in slot i, duplicating the lowest possible numbered house 
                into slot k, and advancing the BIS counter
            -   A (size, 15, size-1) array of fence actions, with each (i,j,k) element representing
                writing house number j+1 in slot i and placing a fence between the k and k+1 spots
            -   A (size, 15, 6) array of estate actions, with each (i,j,k) element representing
                writing house number j+1 in slot i and advancing the k+1-th estate score
            -   A (size, 15) array of park actions with each (i,j) element representing writing
                house number j+1 in slot i and advancing the park counter
            -   A (size, 17) array of temp actions with each (i,j) element representing writing
                house number j+1 in slot i and advancing the temp counter
            -   A (3, 15) array of pool actions with each (i,j) element representing writing house
                number j+1 in the i-th pool slot, in addition to circling the pool and advancing the
                pool counter
            -   A (size, 15) array of house only actions with each (i,j) element representing
                writing house number j+1 in slot i and nothing else
        '''

        house_actions = size * MAX_HOUSE_NUMBER
        temp_actions = size * MAX_TEMP_NUMBER
        self.NUM_BIS_ACTIONS = house_actions * size
        pool_actions = MAX_HOUSE_NUMBER * self.NUM_POOLS
        fence_actions = MAX_HOUSE_NUMBER * self.NUM_FENCES
        park_actions = MAX_HOUSE_NUMBER
        estate_actions = MAX_HOUSE_NUMBER * estate_actions_view.size
        house_only_actions = house_actions

        self.num_actions = self.NUM_BIS_ACTIONS + fence_actions + \
            estate_actions + park_actions + temp_actions + pool_actions + house_only_actions

        # self.null_action = np.zeros(self.num_actions, dtype=bool)
        # self.null_action[-1] = True

        # # self.null_house_actions = self.broadcast_house_actions(
        # #     np.zeros(self.num_house_actions, dtype=bool))
        # self.null_bis_actions = np.zeros((size, MAX_HOUSE_NUMBER), dtype=bool)
        # self.null_fence_actions = np.zeros(fence_actions, dtype=bool)

        # self.bis_neighbors = np.zeros(
        #     (size, MAX_HOUSE_NUMBER, size, MAX_HOUSE_NUMBER), dtype=bool)
        # self.NEIGHBOR_DIAGS = np.eye(
        #     size, size, 1, dtype=bool) + np.eye(size, size, -1, dtype=bool)
        # idx = np.ogrid[:size, :MAX_HOUSE_NUMBER]
        # self.bis_neighbors[idx[0], idx[1], :,
        #                    idx[1]] = self.NEIGHBOR_DIAGS[idx[0]]
        # self.bis_neighbor_mask = np.ones(
        #     (size, 1, size, MAX_HOUSE_NUMBER), dtype=bool)
        # self.bis_neighbor_mask[idx[0], 0, idx[0], :] = False
        pass

    def reset(self):
        self.houses = np.zeros(self.SIZE, dtype=np.uint8)
        self.fences = np.zeros(self.NUM_FENCES, dtype=bool)
        self.pools = np.zeros(self.NUM_POOLS, dtype=bool)
        self.parks = 0
        self.valid_ranges = np.ones((self.SIZE, MAX_HOUSE_NUMBER), dtype=bool)

    def get_observation(self):
        return {
            "houses": self.houses,
            "fences": self.fences,
            "pools": self.pools,
            "parks": self.parks
        }

    def get_action_mask(self, house_number: int, action_card: ActionCard) -> Optional[np.ndarray[bool]]:
        new_house_nums = Row.house_number_diag[house_number]
        if action_card == ActionCard.TEMP:
            new_house_nums[(max(house_number - 2, 0)):(min(house_number + 3, MAX_HOUSE_NUMBER))] = True
        house_actions_mask = np.logical_and(self.valid_ranges, new_house_nums)
        if not house_actions_mask.any():
            return None

        if action_card == ActionCard.POOL:
            pool_actions = house_actions_mask[self.POOL_IDX, :]
        else:
            pool_actions = np.zeros(
                (self.NUM_POOLS, MAX_HOUSE_NUMBER), dtype=bool)

        valid_house_actions = self.create_house_actions(action_card)

        house_actions = np.where(
            house_actions_mask[..., np.newaxis], valid_house_actions, False)

        return np.concatenate((house_actions.ravel(), pool_actions.ravel(), [False]))

    # def apply_action(self, action: int):

    def create_house_actions(self, action_card: ActionCard) -> np.ndarray[bool]:
        park_action = action_card == ActionCard.PARK and self.parks < self.NUM_PARKS
        temp_action = action_card == ActionCard.TEMP
        fence_actions = self.null_fence_actions if action_card != ActionCard.FENCE else np.invert(
            self.fences)
        estate_actions = Row.null_estate_actions if action_card != ActionCard.ESTATE else self.estate_actions
        house_only_action = True
        non_bis_actions = np.concatenate((fence_actions, estate_actions, [
                                         park_action], [temp_action], [house_only_action]))
        if action_card != ActionCard.BIS:
            return np.concatenate((np.zeros(self.NUM_BIS_ACTIONS, dtype=bool), non_bis_actions))[np.newaxis, np.newaxis, :]

        return np.concatenate((self.generate_bis_actions().reshape((self.SIZE, MAX_HOUSE_NUMBER, -1)),
                               np.broadcast_to(non_bis_actions, (self.SIZE, MAX_HOUSE_NUMBER, non_bis_actions.size))), axis=2)

    def generate_bis_actions(self) -> np.ndarray[bool]:
        empty_slots = self.houses == 0
        house_mask = np.logical_and(self.bis_neighbor_mask, empty_slots[
            np.newaxis, np.newaxis, :, np.newaxis])
        expanded_house_map = Row.house_number_diag_one[self.houses]
        neighbors_xor = np.pad(np.logical_xor(
            expanded_house_map[1:, :], expanded_house_map[:-1, :]), ((1, 1), (0, 0)))
        current_neighbors = np.logical_or(
            neighbors_xor[1:, :], neighbors_xor[:-1, :])
        return np.logical_and(np.logical_or(self.bis_neighbors, current_neighbors[np.newaxis, np.newaxis, ...]), house_mask)

    def broadcast_house_actions(self, house_actions: np.ndarray[bool]) -> np.ndarray[bool]:
        return np.broadcast_to(house_actions, (self.SIZE, MAX_HOUSE_NUMBER, self.num_house_actions))


def do_stuff():
    r = Row(10, [2, 6, 7], 4, np.ones(6, dtype=bool))
    r2 = Row(11, [2, 6, 7], 5, np.ones(6, dtype=bool))
    r3 = Row(12, [2, 6, 7], 6, np.ones(6, dtype=bool))
    sum = r.num_actions + r2.num_actions + r3.num_actions
    r.reset()
    r.houses[2] = 4
    r.houses[3] = 4
    # r.get_action_mask(5, ActionCard.POOL)
    return r.get_action_mask(5, ActionCard.BIS)


do_stuff()
