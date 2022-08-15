import cv2 as cv
import os
import torch
from time import time, sleep
from argparse import ArgumentParser
from typing import Optional, Tuple, List
from copy import copy
from datetime import datetime
from enum import Enum, auto
from math import sqrt

FILENAME: Optional[str] = None
URL: Optional[str] = r'rtsp://admin:admin@10.98.26.30:554/live/main'


class Direction(Enum):
    down_left = auto()
    left = auto()
    up_left = auto()
    up = auto()
    up_right = auto()
    right = auto()
    down_right = auto()
    down = auto()
    center = auto()


class CubeState(Enum):
    no_cube = auto()
    cube_arrives = auto()
    cube_is_wrapping = auto()
    wrapped_cube_leaves = auto()


class Rectangle:
    _x: int
    _y: int
    _x2: int
    _y2: int

    def __init__(self, x: int, y: int, x2: int, y2: int) -> None:
        if x > x2 or y > y2:
            self._x = x2
            self._y = y2
            self._x2 = x
            self._y2 = y
        else:
            self._x2 = x2
            self._y2 = y2
            self._x = x
            self._y = y

    def get_adjacent_pairs_of_points(self) -> (
        List[
            Tuple[
                Tuple[int, int],
                Tuple[int, int]
            ]
        ]
    ):
        x, y, x2, y2, x3, y3, x4, y4 = self.get_four_points()
        first = x, y
        second = x2, y2
        third = x3, y3
        fourth = x4, y4
        return [(first, third), (first, fourth), (second, third), (second, fourth)]

    def get_up_left_point(self) -> Tuple[int, int]:
        return self._x, self._y

    def get_down_right_point(self) -> Tuple[int, int]:
        return self._x2, self._y2

    def get_two_points(self) -> Tuple[int, int, int, int]:
        return self._x, self._y, self._x2, self._y2

    def get_four_points(self) -> Tuple[int, int, int, int, int, int, int, int]:
        return self._x, self._y, self._x2, self._y2, self._x, self._y2, self._x2, self._y

    def get_rectangle_area(self):
        return abs(self._x2 - self._x) * abs(self._y2 - self._y)

    def get_copy(self):
        return copy(self)

    def has_inside(self, point: Tuple[int, int]) -> bool:
        return self._x2 >= point[0] >= self._x and self._y2 >= point[1] >= self._y

    def apply_offset_to_points(self, x: int, y: int, x2: int, y2: int):
        self._x += x
        self._y += y
        self._x2 += x2
        self._y2 += y2


class CONST:
    cube_size_mod = int(1e2)
    cube_max_no_move_px = 10 # TODO

    wrapper_min_area = int(1e4)
    wrapper_circles_for_one_wrap = 10

    one_second = 1
    camera_reconnect_sec = 2
    camera_connect_lost_sec = 10
    multiple_cubes_err_delay_sec = 5

    cam_id: str = 'cube_wrap'

    camera_reconnect_text = 'Переподключение к камере'
    camera_connect_lost_text = 'Пропало подключение к камере'

    debug_root_dirname = 'debug'
    debug_wrapper_imgs_dirname = 'wrapper_moves'

    debug_show_vid = False
    debug_show_time_in_console = False
    debug_show_left_border = False
    debug_skip_sec_beginning = 90
    debug_skip_sec = 10

    # Сделать метод для склеивания этих статусов и get_str_from_direction
    status_cube_was_wrapped = 'Куб обмотан'
    status_multiple_cubes = 'Обнаружено более одного куба!'

    log_root_dirname = 'logs'
    log_status_error = 0
    log_status_event = 1

    @classmethod
    def status_cube_arrived(cls, direction: Optional[Direction] = None):
        main_message = 'Куб выехал'
        if direction is None:
            return main_message
        return f'{main_message}: {cls.get_str_from_direction(direction, arrival=True)}'

    @classmethod
    def status_cube_left(cls, wrapped: bool = True, direction: Optional[Direction] = None):
        main_message = 'Куб уехал' if wrapped else 'Необмотанный куб уехал'
        if direction is None:
            return main_message
        return f'{main_message}: {cls.get_str_from_direction(direction, arrival=False)}'

    @classmethod
    def get_str_from_direction(cls, direction: Direction, arrival: bool) -> str:
        if direction is Direction.down_left:
            if arrival:
                return 'снизу слева'
            return 'вниз влево'
        elif direction is Direction.left:
            if arrival:
                return 'слева'
            return 'влево'
        elif direction is Direction.up_left:
            if arrival:
                return 'сверху слева'
            return 'вверх влево'
        elif direction is Direction.up:
            if arrival:
                return 'сверху'
            return 'вверх'
        elif direction is Direction.up_right:
            if arrival:
                return 'сверху справа'
            return 'вверх вправо'
        elif direction is Direction.right:
            if arrival:
                return 'справа'
            return 'вправо'
        elif direction is Direction.down_right:
            if arrival:
                return 'снизу справа'
            return 'вниз вправо'
        elif direction is Direction.down:
            if arrival:
                return 'снизу'
            return 'вниз'
        elif direction is Direction.center:
            return 'по центру'

    @classmethod
    def get_log_line(cls, date_time: str, event: str, cam_id: str, text: str):
        return date_time + ' | ' + event + ' | ' + cam_id + ' | ' + text + '\n'

    @classmethod
    def get_log_filename(cls, date):
        return f'./{cls.log_root_dirname}/' + date + '.txt'

    @classmethod
    def get_debug_circle_full_dirname(cls, cube_full_dirname: str, circle_i: int):
        return f'{cube_full_dirname}/{cls.debug_wrapper_imgs_dirname}/circle_{circle_i}'

    @classmethod
    def get_debug_cube_full_dirname(
        cls,
        vid_name: str,
        frame_n: int,
        cube_was_wrapped: bool
    ) -> str:
        first_part = f'{cls.debug_root_dirname}/{vid_name}_{frame_n}_cube'
        postfix = '_wrapped' if cube_was_wrapped else '_not_wrapped'
        return first_part + postfix

    @classmethod
    def get_cube_stop_png_name(cls, frame_n: int, sec: int):
        return f'cube_{frame_n}_{sec}.png'

    @classmethod
    def get_wrapper_png_name(cls, frame_n: int, sec: int):
        return f'{frame_n}_{sec}.png'

    @classmethod
    def apply_cube_size_modifier(cls, square: Rectangle, with_minus=False):
        sign = 1 if not with_minus else -1
        square.apply_offset_to_points(
            sign * -cls.cube_size_mod,
            sign * -cls.cube_size_mod,
            sign * cls.cube_size_mod,
            sign * cls.cube_size_mod
        )

    @classmethod
    def print_log(cls, text: str, event_n: int):
        print(datetime.today(), text)
        cls.write_log(text, event_n)

    @classmethod
    def write_log(cls, text: str, event_n: int):
        event_str = 'EVENT' if event_n == cls.log_status_event else 'ERROR'
        date_time = datetime.now().astimezone()
        file_name = cls.get_log_filename(str(date_time.date()))
        os.makedirs(cls.log_root_dirname, exist_ok=True)
        with open(file_name, 'a') as log:
            log.write(cls.get_log_line(
                str(date_time), event_str, cls.cam_id, text)
            )


class WrapperInfo:
    frame_n: int
    sec: int
    rectangle: Rectangle

    def __init__(self, frame_n: int, sec: int, rectangle: Rectangle):
        self.frame_n = frame_n
        self.sec = sec
        self.rectangle = rectangle


class CubeInfo:
    cube_stop_frame_n: int
    cube_stop_sec: int
    square: Rectangle

    was_wrapped: bool = False

    def __init__(self, cube_stop_frame_n: int, cube_stop_sec: int, square: Rectangle) -> None:
        self.cube_stop_frame_n = cube_stop_frame_n
        self.cube_stop_sec = cube_stop_sec
        self.square = square


class WrapCircle:
    cube_square: Rectangle
    _diagonal_1: Tuple[float, float]  # y = K1 * x + B1
    _diagonal_2: Tuple[float, float]  # y = K2 * x + B2
    _down: Optional[WrapperInfo] = None
    _left: Optional[WrapperInfo] = None
    _up: Optional[WrapperInfo] = None
    _right: Optional[WrapperInfo] = None

    def __init__(self, cube_square: Rectangle) -> None:
        self.cube_square = cube_square

        x, y, x2, y2, x3, y3, x4, y4 = cube_square.get_four_points()
        d1_a, d1_b, d1_c = (y - y2, x2 - x, x*y2 - x2*y)
        d2_a, d2_b, d2_c = (y3 - y4, x4 - x3, x3*y4 - x4*y3)
        self._diagonal_1 = (-d1_a/d1_b, -d1_c/d1_b)
        self._diagonal_2 = (-d2_a/d2_b, -d2_c/d2_b)

    def try_get_circle(self) -> Optional[List[WrapperInfo]]:
        circle = [self._down, self._left, self._up, self._right]
        if None not in circle:
            return circle

    def try_add_wrapper_info(self, wrapper_info: WrapperInfo) -> bool:
        wrapper_rect = wrapper_info.rectangle
        return (
            self._wrapper_and_cube_collide(wrapper_rect)
            and self._wr_info_has_unique_sec(wrapper_info)
            and (
                self._try_set_down(wrapper_info)
                or self._try_set_left(wrapper_info)
                or self._try_set_up(wrapper_info)
                or self._try_set_right(wrapper_info)
            )
        )

    def _wr_info_has_unique_sec(self, wrapper_info: WrapperInfo) -> bool:
        sec = wrapper_info.sec
        return not (
            (self._down is not None
             and sec == self._down.sec)
            or (self._up is not None
                and sec == self._up.sec)
            or (self._left is not None
                and sec == self._left.sec)
            or (self._right is not None
                and sec == self._right.sec)
        )

    def _wrapper_and_cube_collide(self, wrapper_rect: Rectangle) -> bool:
        x, y, x2, y2, x3, y3, x4, y4 = wrapper_rect.get_four_points()
        for point in [[x, y], [x2, y2], [x3, y3], [x4, y4]]:
            if self.cube_square.has_inside(point):
                return True
        return False

    def _try_set_down(self, wrapper_info: WrapperInfo) -> bool:
        if self._down is not None:
            return False

        x, y, x2, y2, x3, y3, x4, y4 = wrapper_info.rectangle.get_four_points()
        for point in [[x, y], [x2, y2], [x3, y3], [x4, y4]]:
            if not self._point_is_down(point):
                return False
        else:
            self._down = wrapper_info
            return True

    def _point_is_down(self, point: Tuple[int, int]):
        d1_y, d2_y = self._get_points_y_on_diagonals(point[0])
        return point[1] >= d1_y and point[1] >= d2_y

    def _try_set_left(self, wrapper_info: WrapperInfo) -> bool:
        if self._left is not None:
            return False

        x, y, x2, y2, x3, y3, x4, y4 = wrapper_info.rectangle.get_four_points()
        for point in [[x, y], [x2, y2], [x3, y3], [x4, y4]]:
            if not self._point_is_left(point):
                return False
        else:
            self._left = wrapper_info
            return True

    def _point_is_left(self, point: Tuple[int, int]):
        d1_y, d2_y = self._get_points_y_on_diagonals(point[0])
        return point[1] <= d1_y and point[1] >= d2_y

    def _try_set_up(self, wrapper_info: WrapperInfo) -> bool:
        if self._up is not None:
            return False

        x, y, x2, y2, x3, y3, x4, y4 = wrapper_info.rectangle.get_four_points()
        for point in [[x, y], [x2, y2], [x3, y3], [x4, y4]]:
            if not self._point_is_up(point):
                return False
        else:
            self._up = wrapper_info
            return True

    def _point_is_up(self, point: Tuple[int, int]):
        d1_y, d2_y = self._get_points_y_on_diagonals(point[0])
        return point[1] <= d1_y and point[1] <= d2_y

    def _try_set_right(self, wrapper_info: WrapperInfo) -> bool:
        if self._right is not None:
            return False

        x, y, x2, y2, x3, y3, x4, y4 = wrapper_info.rectangle.get_four_points()
        for point in [[x, y], [x2, y2], [x3, y3], [x4, y4]]:
            if not self._point_is_right(point):
                return False
        else:
            self._right = wrapper_info
            return True

    def _point_is_right(self, point: Tuple[int, int]):
        d1_y, d2_y = self._get_points_y_on_diagonals(point[0])
        return point[1] >= d1_y and point[1] <= d2_y

    def _get_points_y_on_diagonals(self, x: int) -> Tuple[float, float]:
        return (
            self._diagonal_1[0] * x + self._diagonal_1[1],
            self._diagonal_2[0] * x + self._diagonal_2[1]
        )


class CheckCubeWrap_State:
    _fr_counter: int = 0
    _vid_fps: int | float
    _vid_shape: Tuple[int, int]
    _vid_center: Tuple[int, int]

    _cube_state: CubeState = CubeState.no_cube
    _cube: Optional[Rectangle] = None
    _cube_info: Optional[CubeInfo] = None
    _cube_circles: Optional[List[WrapCircle]] = None

    _last_multiple_cubes_time: Optional[float] = None
    _last_cube_frame_n: Optional[int] = None
    _last_cube: Optional[Rectangle] = None
    _last_cube_info: Optional[CubeInfo] = None
    _last_cube_circles: Optional[List[WrapCircle]] = None

    _first_cube_moved: bool = False

    def __init__(self, vid) -> None:
        self._vid_fps = vid.get(cv.CAP_PROP_FPS)
        self._vid_shape = (
            int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
        self._vid_center = (self._vid_shape[0]//2, self._vid_shape[1]//2)

    def signal_no_cube(self, print_log: bool = True):
        if self._cube_state is CubeState.no_cube:
            return
        if print_log:
            direction = self.get_direction_from_rect(self._last_cube)
            if (
                self._cube_state is CubeState.cube_is_wrapping
                or self._cube_state is CubeState.cube_arrives
            ):
                CONST.print_log(
                    CONST.status_cube_left(wrapped=False, direction=direction),
                    CONST.log_status_error
                )
            elif self._cube_state is CubeState.wrapped_cube_leaves:
                status = (
                    CONST.log_status_error
                    if direction is Direction.center
                    else CONST.log_status_event
                )
                CONST.print_log(
                    CONST.status_cube_left(direction=direction),
                    status
                )
        self._fr_counter = 0

        self._cube_state = CubeState.no_cube
        self._cube = None
        self._cube_info = None
        self._cube_circles = None

    def signal_cube_stands(self):
        if self._cube_state is CubeState.cube_arrives:
            self._cube_state = CubeState.cube_is_wrapping

        cube = self._cube.get_copy()
        CONST.apply_cube_size_modifier(cube)
        cube_stop_frame_n: int = (
            self.get_frame_n_for_info(self._last_cube_frame_n)
        )
        self._cube_info = CubeInfo(
            cube_stop_frame_n=cube_stop_frame_n,
            cube_stop_sec=int(cube_stop_frame_n / self._vid_fps),
            square=cube
        )
        if self._cube_circles is None:
            self._cube_circles = []

        self._last_cube_info = self._cube_info
        self._last_cube_circles = self._cube_circles

    def signal_cube_moves(self, next_cube: Rectangle):
        if self._cube_state is CubeState.no_cube:
            direction = self.get_direction_from_rect(next_cube)
            status = (
                CONST.log_status_error
                if direction is not Direction.right
                else CONST.log_status_event
            )
            if not self._first_cube_moved:
                status = CONST.log_status_event
                self._first_cube_moved = True
            CONST.print_log(
                CONST.status_cube_arrived(direction=direction),
                status
            )
            self._cube_state = CubeState.cube_arrives

        self._cube = next_cube.get_copy()

        self._last_cube = self._cube
        self._last_cube_frame_n = self._fr_counter

    def signal_cube_wrapped(self):
        CONST.print_log(CONST.status_cube_was_wrapped, CONST.log_status_event)
        self._cube_info.was_wrapped = True
        self._cube_state = CubeState.wrapped_cube_leaves

    def set_last_multiple_cubes_time(self):
        self._last_multiple_cubes_time = time()

    def increment_fr_counter(self):
        self._fr_counter += 1

    def get_last_cube_info(self) -> Optional[Tuple[CubeInfo, List[WrapCircle]]]:
        return (
            None
            if self._last_cube_info is None
            else (self._last_cube_info, self._last_cube_circles)
        )

    def get_frame_n_for_info(self, frame_n: int):
        return (
            int(frame_n + self._vid_fps * CONST.debug_skip_sec_beginning)
            if CONST.debug_show_vid
            else frame_n
        )

    def get_dist_to_vid_center(self, rect: Rectangle):
        x, y, x2, y2 = rect.get_two_points()
        v = ((x2 - x)/2, (y2 - y)/2)
        xc, yc = int(x + v[0]), int(y + v[1])
        return self.get_dist_between_two_points(
            xc,
            yc,
            self._vid_center[0],
            self._vid_center[1]
        )

    def get_direction_from_rect(self, rect: Rectangle) -> Direction:
        points = rect.get_four_points()
        first = points[0], points[1]
        second = points[2], points[3]
        third = points[4], points[5]
        fourth = points[6], points[7]
        ordered_points = first, fourth, second, third
        quadrants = [self.get_point_quadrant(*i) for i in ordered_points]
        if quadrants[0] == quadrants[1] == quadrants[2] == quadrants[3] == 1:
            return Direction.up_right
        if quadrants[0] == quadrants[1] == quadrants[2] == quadrants[3] == 2:
            return Direction.up_left
        if quadrants[0] == quadrants[1] == quadrants[2] == quadrants[3] == 3:
            return Direction.down_left
        if quadrants[0] == quadrants[1] == quadrants[2] == quadrants[3] == 4:
            return Direction.down_right

        if quadrants[0] == 2 and quadrants[1] == 1 and quadrants[2] == 4 and quadrants[3] == 3:
            return Direction.center

        if quadrants[0] == quadrants[1] == 1 and quadrants[2] == quadrants[3] == 4:
            return Direction.right
        if quadrants[0] == quadrants[1] == 2 and quadrants[2] == quadrants[3] == 3:
            return Direction.left

        if quadrants[0] == quadrants[3] == 2 and quadrants[1] == quadrants[2] == 1:
            return Direction.up
        if quadrants[0] == quadrants[3] == 3 and quadrants[1] == quadrants[2] == 4:
            return Direction.down

    def get_point_quadrant(self, x: int, y: int) -> int:
        if x > self._vid_center[0] and y > self._vid_center[1]:
            return 4
        elif x < self._vid_center[0] and y > self._vid_center[1]:
            return 3
        elif x > self._vid_center[0] and y < self._vid_center[1]:
            return 1
        return 2

    def last_multiple_cubes_time(self):
        return self._last_multiple_cubes_time

    def fr_counter(self):
        return self._fr_counter

    def vid_fps(self):
        return self._vid_fps

    def cube(self):
        return self._cube

    def last_cube_frame_n(self):
        return self._last_cube_frame_n

    def cube_is_wrapping(self):
        return self._cube_state is CubeState.cube_is_wrapping

    def cube_moved(self, next_cube: Rectangle) -> bool: # TODO
        if self._cube is None:
            return True
        first = next_cube.get_two_points()[:2]
        second = self._cube.get_two_points()[:2]
        dist = self.get_dist_between_two_points(*first, *second)
        if dist > CONST.cube_max_no_move_px:
            return True
        return False

    def one_sec_without_move(self):
        return (
            self._last_cube_frame_n is not None
            and (self._fr_counter - self._last_cube_frame_n) / self._vid_fps >= CONST.one_second
        )

    def all_circles_are_done(self):
        return (
            len(self._cube_circles) == CONST.wrapper_circles_for_one_wrap
            and self._cube_circles[-1].try_get_circle() is not None)

    def try_add_wrapper_info(self, wrapper_info: WrapperInfo) -> bool:
        if (
            len(self._cube_circles) == 0
            or self._cube_circles[-1].try_get_circle() is not None
        ):
            self._cube_circles.append(
                WrapCircle(cube_square=self._cube_info.square)
            )
        return self._cube_circles[-1].try_add_wrapper_info(wrapper_info)

    @staticmethod
    def get_dist_between_two_points(x, y, x2, y2):
        return sqrt((x2 - x)**2 + (y2 - y)**2)


class CheckCubeWrap:
    _state: CheckCubeWrap_State

    _vid_name: str
    _vid_cam: str

    _model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='cube_wrap_detect.pt',
        _verbose=False
    )

    def check_cube_wrap(self, vid_name):
        if FILENAME is not None:
            self._check_cube_wrap_vid(FILENAME)
        elif vid_name is not None and os.path.isfile(vid_name):
            self._check_cube_wrap_vid(vid_name)
        elif URL is not None:
            self._check_cube_wrap_cam(URL)
        elif vid_name is not None:
            self._check_cube_wrap_cam(vid_name)

    def _check_cube_wrap_vid(self, vid_name):
        self._vid_name = vid_name
        vid = cv.VideoCapture(self._vid_name)
        self._state = CheckCubeWrap_State(vid)
        state = self._state
        if CONST.debug_show_vid:
            vid.set(1, state.vid_fps() * CONST.debug_skip_sec_beginning)
        ret, frame1 = self._read_frame(vid)
        ret, frame2 = self._read_frame(vid)
        if CONST.debug_show_time_in_console:
            timer = time()
        while ret and vid.isOpened():
            if state.fr_counter() % int(state.vid_fps() * CONST.one_second / 2) == 0:
                self._process_contours(frame1, frame2)

            if CONST.debug_show_vid:
                self._paint_vid(frame1)
                frame1 = cv.resize(frame1, dsize=None, fx=0.5, fy=0.5)
                cv.imshow("", frame1)
                waitkey = cv.waitKey(1)
                if waitkey == ord('q'):
                    break
                elif waitkey == ord('d'):
                    for _ in range(int(state.vid_fps() * CONST.debug_skip_sec)):
                        state.increment_fr_counter()
                    vid.set(
                        1,
                        CONST.debug_skip_sec_beginning * state.vid_fps() + state.fr_counter()
                    )

            frame1 = frame2
            ret, frame2 = self._read_frame(vid)
            sec = state.fr_counter() / int(state.vid_fps())

            if CONST.debug_show_time_in_console and sec % 60 == 0 and sec // 60 != 0:
                print(
                    f'{int(sec//60)} min / {int(time()-timer)} sec'
                )
        self._end_the_check_vid(vid)

    def _check_cube_wrap_cam(self, vid_cam):  # TODO
        self._vid_cam = vid_cam
        vid = cv.VideoCapture(vid_cam)
        self._state = CheckCubeWrap_State(vid)
        ret, frame1 = self._try_read_frame_cam(vid)
        ret, frame2 = self._try_read_frame_cam(vid)
        while ret and vid.isOpened():
            self._process_contours(frame1, frame2)

            frame1 = frame2
            ret, frame2 = self._try_read_frame_cam(vid)

        self._end_the_check_cam(vid)

    def _try_read_frame_cam(self, vid):
        connect_lost = False
        while True:
            try:
                ret, frame = self._read_frame(vid)
                if not ret:
                    if connect_lost:
                        raise
                    connect_lost = True

                    vid.release()
                    CONST.write_log(
                        CONST.camera_reconnect_text,
                        CONST.log_status_error
                    )
                    sleep(CONST.camera_reconnect_sec)
                    vid = cv.VideoCapture(self._vid_cam)
                    continue
                return ret, frame
            except:
                CONST.write_log(
                    CONST.camera_connect_lost_text,
                    CONST.log_status_error
                )
                sleep(CONST.camera_connect_lost_sec)
                self._state.signal_no_cube(print_log=False)
                connect_lost = False

    def _read_frame(self, vid):
        self._state.increment_fr_counter()
        return vid.read()

    def _paint_vid(self, frame1):
        state = self._state
        if state.cube() is not None:
            cv.rectangle(
                frame1,
                state.cube().get_up_left_point(),
                state.cube().get_down_right_point(),
                (255, 0, 0),
                thickness=2
            )
        if state.get_last_cube_info() is not None:
            cube, _ = state.get_last_cube_info()
            cv.rectangle(
                frame1,
                cube.square.get_up_left_point(),
                cube.square.get_down_right_point(),
                (0, 255, 0),
                thickness=2
            )

    def _process_contours(self, frame1, frame2):
        state = self._state
        res = self._model(cv.cvtColor(frame1, cv.COLOR_BGR2RGB), size=640)
        df = res.pandas().xyxy[0]
        cube = None
        if len(df.index) == 1:
            cube = Rectangle(
                int(df['xmin'][0]),
                int(df['ymin'][0]),
                int(df['xmax'][0]),
                int(df['ymax'][0])
            )
        elif len(df.index) > 1:
            if (
                state.last_multiple_cubes_time() is not None
                and time() - state.last_multiple_cubes_time() > CONST.multiple_cubes_err_delay_sec
            ):
                CONST.print_log(CONST.status_multiple_cubes,
                                CONST.log_status_error)
                state.set_last_multiple_cubes_time()
            cubes = []
            dists = []
            for i in range(len(df.index)):
                cubes.append(
                    Rectangle(
                        int(df['xmin'][0]),
                        int(df['ymin'][0]),
                        int(df['xmax'][0]),
                        int(df['ymax'][0])
                    )
                )
                dists.append(state.get_dist_to_vid_center(cubes[i]))
            min_i = dists.index(min(dists))
            cube = cubes[min_i]

        self._check_cube(cube)
        if state.cube_is_wrapping():
            for contour in self._get_countours(frame1, frame2):
                (x, y, w, h) = cv.boundingRect(contour)
                x2, y2 = x+w, y+h
                if w * h >= CONST.wrapper_min_area:
                    self._check_wrapper(x, y, x2, y2)
                    if CONST.debug_show_vid:
                        cv.rectangle(
                            frame1,
                            (x, y),
                            (x2, y2),
                            (0, 0, 255),
                            thickness=2)

    def _get_countours(self, frame1, frame2):
        return cv.findContours(
            self._get_frames_diff(frame1, frame2),
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)[0]

    def _check_wrapper(self, x, y, x2, y2):
        state = self._state
        wrapper_info = WrapperInfo(
            frame_n=state.get_frame_n_for_info(state.fr_counter()),
            sec=int(state.fr_counter()/state.vid_fps()),
            rectangle=Rectangle(x=x, y=y, x2=x2, y2=y2)
        )
        if state.try_add_wrapper_info(wrapper_info) and state.all_circles_are_done():
            state.signal_cube_wrapped()

    def _check_cube(self, cube: Optional[Rectangle]):
        state = self._state
        if cube is None:
            state.signal_no_cube()
        elif state.cube_moved(cube):
            state.signal_cube_moves(cube)
        elif state.one_sec_without_move():
            state.signal_cube_stands()

    def _end_the_check_cam(self, vid_cam):
        vid_cam.release()
        if CONST.debug_show_vid:
            cv.destroyAllWindows()

    def _end_the_check_vid(self, vid):
        if self._state.get_last_cube_info() is not None:
            cube_info, wrapper_circles = self._state.get_last_cube_info()
            self._write_cube_info(cube_info)
            self._write_wrapper_circles(cube_info, wrapper_circles)
            vid.release()
        if CONST.debug_show_vid:
            cv.destroyAllWindows()

    def _write_cube_info(self, cube_info: CubeInfo):
        CONST.apply_cube_size_modifier(
            cube_info.square,
            with_minus=True
        )
        cube_full_dirname = CONST.get_debug_cube_full_dirname(
            self._vid_name,
            cube_info.cube_stop_frame_n,
            cube_info.was_wrapped)
        os.makedirs(f'{cube_full_dirname}', exist_ok=True)
        self._create_cube_stop_img(cube_info, cube_full_dirname)

    def _write_wrapper_circles(
        self,
        cube_info: CubeInfo,
        wrapper_circles: List[WrapCircle]
    ):
        circle_i = 1
        cube_full_dirname = CONST.get_debug_cube_full_dirname(
            self._vid_name,
            cube_info.cube_stop_frame_n,
            cube_info.was_wrapped
        )
        for circle in wrapper_circles:
            circle_full_dirname = (
                CONST.get_debug_circle_full_dirname(cube_full_dirname, circle_i))
            os.makedirs(circle_full_dirname, exist_ok=True)
            wrapper_infos = circle.try_get_circle()
            if wrapper_infos is not None:
                for wrapper_info in wrapper_infos:
                    self._create_wrapper_stop_img(
                        circle_full_dirname, wrapper_info
                    )
            circle_i += 1

    def _create_wrapper_stop_img(self, dirname: str, wrapper_info: WrapperInfo):
        wrapper_frame = self._cut_frame_from_vid(wrapper_info.frame_n)
        wrapper_img_filename = (
            CONST.get_wrapper_png_name(wrapper_info.frame_n, wrapper_info.sec)
        )
        cv.rectangle(
            wrapper_frame,
            wrapper_info.rectangle.get_up_left_point(),
            wrapper_info.rectangle.get_down_right_point(),
            (255),
            thickness=2
        )
        cv.imwrite(f'{dirname}/{wrapper_img_filename}', wrapper_frame)

    def _create_cube_stop_img(self, cube_info: CubeInfo, cube_full_dirname: str):
        cube_stop_frame = (
            self._cut_frame_from_vid(cube_info.cube_stop_frame_n)
        )
        cube_stop_filename = (
            CONST.get_cube_stop_png_name(
                cube_info.cube_stop_frame_n, cube_info.cube_stop_sec)
        )
        cv.rectangle(
            cube_stop_frame,
            cube_info.square.get_up_left_point(),
            cube_info.square.get_down_right_point(),
            (255),
            thickness=2
        )
        cv.imwrite(
            f'{cube_full_dirname}/{cube_stop_filename}', cube_stop_frame)

    def _cut_frame_from_vid(self, frame_n: int):
        vid = cv.VideoCapture(self._vid_name)
        vid.set(1, frame_n)
        _, frame = vid.read()
        vid.release()
        return frame

    @staticmethod
    def _get_frames_diff(frame1, frame2):
        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        return dilated


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('vid_stream', type=str, nargs='?')
    args = arg_parser.parse_args()

    CheckCubeWrap().check_cube_wrap(args.vid_stream)
