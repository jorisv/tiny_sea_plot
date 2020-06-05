import math
import typing

from pytinysea import (
    time_t,
    meter_t,
    latitude_t,
    longitude_t,
    NVector,
    TimeWorldMap,
    BoatVelocityTable,
)
from pytinysea.gsp import (
    State,
    BinaryHeapOpenList,
    CloseList,
    NeighborsFinder,
    StateFactory,
    Result,
    find_global_shortest_path,
)


def compute_shortest_path(
    time_world_map: TimeWorldMap,
    boat_velocity_table: BoatVelocityTable,
    start_pos: typing.Tuple[latitude_t, longitude_t],
    end_pos: typing.Tuple[latitude_t, longitude_t],
    start_time: time_t,
):
    start_nvector = NVector.from_lat_lon(*start_pos)
    end_nvector = NVector.from_lat_lon(*end_pos)

    discret_distance_step = 1000.0
    discret_distance_step_max_size = math.sqrt(2 * (discret_distance_step ** 2)) + 1.0

    state_factory = StateFactory(
        time_t(60.0 * 10.0),
        meter_t(discret_distance_step),
        meter_t(6371.0 * 1e3),
        end_nvector,
        boat_velocity_table.max_velocity(),
    )

    open_list = BinaryHeapOpenList()
    open_list.insert(state_factory.build(start_nvector, start_time))

    close_list = CloseList()

    neighbors_finder = NeighborsFinder(
        state_factory,
        time_world_map,
        boat_velocity_table,
        meter_t(discret_distance_step_max_size),
    )
    res = find_global_shortest_path(
        state_factory.build(end_nvector, time_t(0.0)),
        open_list,
        close_list,
        neighbors_finder,
    )
    return res, open_list, close_list


def result_to_state_list(
    result: typing.Optional[Result], close_list: CloseList
) -> typing.List[State]:

    if result is None:
        return []

    return state_to_state_list(result.state, close_list)


def state_to_state_list(state: State, close_list: CloseList) -> typing.List[State]:
    cur_state = state
    state_list = [cur_state]
    close_list_store = close_list.store()
    while cur_state.parent_state() is not None:
        cur_state = close_list_store[cur_state.parent_state()]
        state_list.append(cur_state)

    return list(reversed(state_list))
