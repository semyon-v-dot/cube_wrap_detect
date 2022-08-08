import cv2 as cv
import os
from time import time, sleep
from argparse import ArgumentParser
from typing import Any, Optional, Tuple, List
from copy import copy
from datetime import datetime
from enum import Enum, auto

FILENAME: Optional[str] = None
URL: Optional[str] = r'rtsp://admin:admin@10.98.26.30:554/live/main'


class CONST:
    cube_detect_min_area = int(3e5)
    cube_left_border = int(4e2)
    cube_square_size_increase = int(1e2)
    cube_square_x_increase = int(9e1)

    wrapper_min_area = int(1e4)
    wrapper_circles_for_one_wrap = 10

    one_second = 1
    camera_reconnect_sec = 2
    camera_connect_lost_sec = 10

    camera_reconnect_text = 'Переподключение к камере'
    camera_connect_lost_text = 'Пропало подключение к камере'

    debug_root_dirname = 'debug'
    debug_wrapper_imgs_dirname = 'wrapper_moves'

    debug_show_vid = False
    debug_show_time_in_console = False
    debug_show_left_border = False
    debug_skip_sec_beginning = 120
    debug_skip_sec = 10

    status_cube_stands = 'Куб выехал на обмотку'
    status_cube_was_wrapped = 'Куб обмотан'
    status_cube_left = 'Куб уехал'

    log_root_dirname = 'logs'
    log_status_error = 0
    log_status_event = 1

    class Direction(Enum):
        down_left = auto()
        left = auto()
        up_left = auto()
        up = auto()
        up_right = auto()
        right = auto()
        down_right = auto()
        down = auto()

    @classmethod
    def get_str_from_direction(cls, direction: Direction, arrival: bool) -> str:  # TODO
        if direction is CONST.Direction.down_left:
            if arrival:
                return 'снизу слева'
            return 'вниз влево'
        elif direction is CONST.Direction.left:
            if arrival:
                return 'слева'
            return 'влево'
        elif direction is CONST.Direction.up_left:
            if arrival:
                return 'сверху слева'
            return 'вверх влево'
        elif direction is CONST.Direction.up:
            if arrival:
                return 'сверху'
            return 'вверх'
        elif direction is CONST.Direction.up_right:
            if arrival:
                return 'сверху справа'
            return 'вверх вправо'
        elif direction is CONST.Direction.right:
            if arrival:
                return 'справа'
            return 'вправо'
        elif direction is CONST.Direction.down_right:
            if arrival:
                return 'снизу справа'
            return 'вниз вправо'
        elif direction is CONST.Direction.down:
            if arrival:
                return 'снизу'
            return 'вниз'

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
    cube_stands = False
    left_border_was_disturbed = True
    last_b_c_frame_n: Optional[int] = None
    last_big_contour_square: Optional[Rectangle] = None

    _last_cube: Optional[CubeInfo] = None
    _last_cube_circles: Optional[List[WrapCircle]] = None

    def set_cube_info(self, cube_info: CubeInfo):
        self._last_cube = cube_info
        self._last_cube_circles = []

    def set_last_cube_wrapped(self):
        self._last_cube.was_wrapped = True

    def last_cube_was_wrapped(self):
        return self._last_cube.was_wrapped

    def get_last_cube_info(self) -> Optional[Tuple[CubeInfo, List[WrapCircle]]]:
        return None if self._last_cube is None else (self._last_cube, self._last_cube_circles)

    def try_add_wrapper_info(self, wrapper_info: WrapperInfo) -> bool:
        if (
            len(self._last_cube_circles) == 0
            or self._last_cube_circles[-1].try_get_circle() is not None
        ):
            self._last_cube_circles.append(
                WrapCircle(cube_square=self._last_cube.square))
        return self._last_cube_circles[-1].try_add_wrapper_info(wrapper_info)

    def all_circles_are_done(self):
        return (
            len(self._last_cube_circles) == CONST.wrapper_circles_for_one_wrap
            and self._last_cube_circles[-1].try_get_circle() is not None)


class CheckCubeWrap:
    _vid_name: str
    _state = CheckCubeWrap_State()

    _video_cam: str
    _cam_id: str = 'cube_wrap'

    def check_cube_wrap_vid(self, vid_name):
        self._vid_name = vid_name
        vid = cv.VideoCapture(self._vid_name)
        vid_fps = vid.get(cv.CAP_PROP_FPS)
        if CONST.debug_show_vid:
            vid.set(1, vid_fps * CONST.debug_skip_sec_beginning)
        ret, frame1 = vid.read()
        ret, frame2 = vid.read()
        fr_counter = 2
        if CONST.debug_show_time_in_console:
            timer = time()
        while ret and vid.isOpened():
            self._process_contours(frame1, frame2, fr_counter, vid_fps)

            if CONST.debug_show_vid:
                if CONST.debug_show_left_border:
                    cv.line(
                        frame1,
                        (CONST.cube_left_border, 0),
                        (CONST.cube_left_border, 1000),
                        (255)
                    )
                frame1 = cv.resize(frame1, dsize=None, fx=0.5, fy=0.5)
                cv.imshow("", frame1)
                waitkey = cv.waitKey(1)
                if waitkey == ord('q'):
                    break
                elif CONST.debug_show_vid and waitkey == ord('d'):
                    fr_counter += int(vid_fps * CONST.debug_skip_sec)
                    vid.set(
                        1, CONST.debug_skip_sec_beginning * vid_fps + fr_counter
                    )
            frame1 = frame2
            ret, frame2 = vid.read()
            fr_counter += 1
            sec = fr_counter / int(vid_fps)
            if CONST.debug_show_time_in_console and sec % 60 == 0 and sec // 60 != 0:
                print(
                    f'{int(sec//60)} min / {int(time()-timer)} sec'
                )
        self._end_the_check_vid(vid)

    def check_cube_wrap_cam(self, vid_cam):
        self._video_cam = vid_cam
        vid = cv.VideoCapture(vid_cam)
        vid_fps = vid.get(cv.CAP_PROP_FPS)
        ret, frame1 = self._try_read_frame_cam(vid)
        ret, frame2 = self._try_read_frame_cam(vid)
        fr_counter = 2

        while ret and vid.isOpened():
            self._process_contours(frame1, frame2, fr_counter, vid_fps)

            if CONST.debug_show_vid:
                if CONST.debug_show_left_border:
                    cv.line(
                        frame1,
                        (CONST.cube_left_border, 0),
                        (CONST.cube_left_border, 1000),
                        (255)
                    )
                frame1 = cv.resize(frame1, dsize=None, fx=0.5, fy=0.5)
                cv.imshow("", frame1)
                if cv.waitKey(1) == ord('q'):
                    break

            frame1 = frame2
            ret, frame2 = self._try_read_frame_cam(vid)
            fr_counter += 1

        self._end_the_check_cam(vid)

    def _try_read_frame_cam(self, vid):
        connect_lost = False
        while True:
            try:
                ret, frame = vid.read()
                if not ret:
                    if connect_lost:
                        raise
                    connect_lost = True

                    vid.release()
                    self._write_log(
                        CONST.camera_reconnect_text,
                        CONST.log_status_error
                    )
                    sleep(CONST.camera_reconnect_sec)
                    vid = cv.VideoCapture(self._video_cam)
                    continue
                return ret, frame
            except:
                self._write_log(
                    CONST.camera_connect_lost_text,
                    CONST.log_status_error
                )
                sleep(CONST.camera_connect_lost_sec)
                connect_lost = False

    def _process_contours(self, frame1, frame2, fr_counter: int, vid_fps):
        state = self._state
        сontours, _ = cv.findContours(
            self._get_frames_diff(frame1, frame2),
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)
        for contour in сontours:
            (x, y, w, h) = cv.boundingRect(contour)
            x2, y2 = x+w, y+h
            if CONST.debug_show_vid:
                if state.last_big_contour_square is not None:
                    cv.rectangle(
                        frame1,
                        state.last_big_contour_square.get_up_left_point(),
                        state.last_big_contour_square.get_down_right_point(),
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

            self._check_cube(fr_counter, vid_fps, x, y, w, h)
            if (
                state.cube_stands
                and not state.last_cube_was_wrapped()
                and w * h >= CONST.wrapper_min_area
            ):
                self._check_wrapper(fr_counter, vid_fps, x, y, x2, y2)
                if CONST.debug_show_vid:
                    cv.rectangle(
                        frame1,
                        (x, y),
                        (x2, y2),
                        (0, 0, 255),
                        thickness=2)

    def _check_wrapper(self, fr_counter: int, vid_fps, x, y, x2, y2):
        state = self._state
        wrapper_info = WrapperInfo(
            frame_n=self._get_frame_n_for_info(fr_counter, vid_fps),
            sec=int(fr_counter/vid_fps),
            rectangle=Rectangle(x=x, y=y, x2=x2, y2=y2)
        )
        if state.try_add_wrapper_info(wrapper_info) and state.all_circles_are_done():
            self._print_log(CONST.status_cube_was_wrapped,
                            CONST.log_status_event)
            state.set_last_cube_wrapped()

    def _check_cube(self, fr_counter: int, vid_fps, x, y, w, h):
        state = self._state
        x2, y2 = x+w, y+h
        if w * h >= CONST.cube_detect_min_area:
            state.last_b_c_frame_n = fr_counter
            state.last_big_contour_square = (
                Rectangle(x=x2-(y2-y), y=y, x2=x2, y2=y2))
            last_b_c_square_x, _ = state.last_big_contour_square.get_up_left_point()
            state.left_border_was_disturbed = last_b_c_square_x < CONST.cube_left_border
        one_sec_without_b_c = (
            state.last_b_c_frame_n is not None
            and (fr_counter - state.last_b_c_frame_n) / vid_fps >= CONST.one_second)
        if one_sec_without_b_c:
            if not state.left_border_was_disturbed and not state.cube_stands:
                self._print_log(CONST.status_cube_stands,
                                CONST.log_status_event)
                state.cube_stands = True
                self._enlarge_last_b_c_square()
                cube_stop_frame_n: int = (
                    self._get_frame_n_for_info(state.last_b_c_frame_n, vid_fps))
                state.set_cube_info(
                    CubeInfo(
                        cube_stop_frame_n=cube_stop_frame_n,
                        cube_stop_sec=int(cube_stop_frame_n / vid_fps),
                        square=state.last_big_contour_square.get_copy())
                )
            elif state.left_border_was_disturbed:
                self._print_log(CONST.status_cube_left, CONST.log_status_event)
                state.cube_stands = False
            state.last_big_contour_square = None
            state.last_b_c_frame_n = None
            state.left_border_was_disturbed = True

    def _print_log(self, text: str, event_n: int):
        print(datetime.today(), text)
        self._write_log(text, event_n)

    def _get_frame_n_for_info(self, frame_n: int, vid_fps):
        return (
            int(frame_n + vid_fps * CONST.debug_skip_sec_beginning)
            if CONST.debug_show_vid
            else frame_n
        )

    def _end_the_check_cam(self, vid_cam):
        vid_cam.release()
        if CONST.debug_show_vid:
            cv.destroyAllWindows()

    def _end_the_check_vid(self, vid):
        cube_info, wrapper_circles = self._state.get_last_cube_info()
        self._write_cube_info(cube_info)
        self._write_wrapper_circles(cube_info, wrapper_circles)
        vid.release()
        if CONST.debug_show_vid:
            cv.destroyAllWindows()

    def _write_log(self, text: str, event_n: int):
        event_str = 'EVENT' if event_n == CONST.log_status_event else 'ERROR'
        date_time = datetime.now().astimezone()
        file_name = CONST.get_log_filename(str(date_time.date()))
        os.makedirs(CONST.log_root_dirname, exist_ok=True)
        with open(file_name, 'a') as log:
            log.write(CONST.get_log_line(
                str(date_time), event_str, self._cam_id, text))

    def _write_cube_info(self, cube_info: CubeInfo):
        self._downsize_cube_square(cube_info)
        cube_full_dirname = CONST.get_debug_cube_full_dirname(
            self._vid_name,
            cube_info.cube_stop_frame_n,
            cube_info.was_wrapped)
        os.makedirs(f'{cube_full_dirname}', exist_ok=True)
        self._create_cube_stop_img(cube_info, cube_full_dirname)

    def _enlarge_last_b_c_square(self):
        self._apply_cube_square_size_increase(
            self._state.last_big_contour_square
        )

    def _downsize_cube_square(self, cube_info: CubeInfo):
        self._apply_cube_square_size_increase(
            cube_info.square,
            with_minus=True
        )

    def _apply_cube_square_size_increase(self, square: Rectangle, with_minus=False):
        sign = 1 if not with_minus else -1
        square.apply_offset_to_points(
            sign * -(CONST.cube_square_x_increase
                     + CONST.cube_square_size_increase),
            sign * -CONST.cube_square_size_increase,
            sign * CONST.cube_square_size_increase,
            sign * CONST.cube_square_size_increase
        )

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
                        circle_full_dirname, wrapper_info)
            circle_i += 1

    def _create_wrapper_stop_img(self, dirname: str, wrapper_info: WrapperInfo):
        wrapper_frame = self._cut_frame_from_vid(wrapper_info.frame_n)
        wrapper_img_filename = (
            CONST.get_wrapper_png_name(wrapper_info.frame_n, wrapper_info.sec))
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
            self._cut_frame_from_vid(cube_info.cube_stop_frame_n))
        cube_stop_filename = (
            CONST.get_cube_stop_png_name(
                cube_info.cube_stop_frame_n, cube_info.cube_stop_sec))
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

    if FILENAME is not None:
        CheckCubeWrap().check_cube_wrap_vid(FILENAME)
    elif args.vid_stream is not None and os.path.isfile(args.vid_stream):
        CheckCubeWrap().check_cube_wrap_vid(args.vid_stream)
    elif URL is not None:
        CheckCubeWrap().check_cube_wrap_cam(URL)
    elif args.vid_stream is not None:
        CheckCubeWrap().check_cube_wrap_cam(args.vid_stream)
