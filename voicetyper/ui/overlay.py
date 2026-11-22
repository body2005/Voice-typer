"""Production floating pill overlay for VoiceTyper matching the reference design.

Features:
- Glassmorphic dark capsule with state-dependent gradient borders.
- Non-activating window: never steals keyboard focus from active editors/browsers.
- Left-click draggable with persistent coordinates saved to config.json.
- Multi-monitor safety check with fallback above the Windows taskbar clock.
- State 1 (Recording): Glowing red dot with halo + '• LIVE' badge + hybrid visualizer
  (RMS-reactive equalizer bars + edge-tapered dual sine waves) + 3D 'Esc' keycap badge.
- State 2 (Processing): Glowing cyan/teal flowing harmonic wave orb + 'Processing speech...'.
- State 3 (Error): Amber '!' badge + crisp error toast + subtle retry icon.
- State 4 (Standby): Smooth fade down to 0.28 opacity ghost state with 60s auto-hide timeout.
- 60 FPS animation timer runs ONLY when active, stopping in Standby/Hidden to keep CPU at ~0%.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
from ctypes import wintypes
from enum import Enum
from typing import Any

from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# Win32 Window Styles
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

if sys.platform == "win32":
    user32 = ctypes.windll.user32
    GetWindowLongPtr = user32.GetWindowLongPtrW
    SetWindowLongPtr = user32.SetWindowLongPtrW
    GetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int]
    GetWindowLongPtr.restype = ctypes.c_ssize_t
    SetWindowLongPtr.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    SetWindowLongPtr.restype = ctypes.c_ssize_t


class OverlayState(str, Enum):
    """Visual states of the floating pill overlay."""

    HIDDEN = "HIDDEN"
    STANDBY = "STANDBY"
    RECORDING = "RECORDING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"



class BaseOverlayRenderer:
    """Shared visual rendering engine for floating overlay and settings preview."""

    theme: str = "classic"
    pill_width: int = 380
    pill_height: int = 56
    _state: OverlayState = OverlayState.STANDBY
    _pulse_alpha: int = 255
    _pulse_dir: int = -1
    _wave_phase: float = 0.0
    _proc_phase: float = 0.0
    _smoothed_amplitude: float = 0.0
    _num_bars: int = 7
    _bar_heights: list[float] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    _processing_text: str = "Processing speech..."
    _error_text: str = "No speech detected"
    _hover_gear: bool = False
    _is_downloading: bool = False
    _download_pct: float | None = None
    _hover_cancel: bool = False

    def width(self) -> int:
        return self.pill_width

    def height(self) -> int:
        return self.pill_height

    def paint_theme_overlay(self, painter: QPainter, rect: QRectF) -> None:
        """Render overlay content based on self.theme and self._state."""
        if self.theme == "neon_glass":
            self._paint_neon_glass(painter, rect)
        elif self.theme == "matte_island":
            self._paint_matte_island(painter, rect)
        elif self.theme == "fluent_mica":
            self._paint_fluent_mica(painter, rect)
        else:
            self._paint_classic(painter, rect)

    def _paint_classic(self, painter: QPainter, rect: QRectF) -> None:
        """Render Theme 1: Classic floating pill."""
        radius = float(self.pill_height) / 2.0

        # 1. Base Glassmorphic Capsule Background & Ambient Glow
        self._draw_capsule_background(painter, rect, radius)

        # 2. State-Specific Content
        if self._state == OverlayState.RECORDING:
            self._draw_recording_state(painter)
        elif self._state == OverlayState.PROCESSING:
            self._draw_processing_state(painter)
        elif self._state == OverlayState.ERROR:
            self._draw_error_state(painter)
        elif self._state == OverlayState.STANDBY:
            self._draw_standby_state(painter)

        # 3. Quick Settings Gear Icon (⚙) on Far Right & Cancel Button (✕) when downloading
        self._draw_gear_button(painter)
        self._draw_cancel_download_button(painter)

        # 4. State-Specific 1.8px Gradient Border
        self._draw_gradient_border(painter, rect, radius)

    def _draw_capsule_background(self, painter: QPainter, rect: QRectF, radius: float) -> None:
        """Draw semi-transparent dark glassmorphic capsule with soft ambient glow."""
        # Glass fill
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(20, 22, 28, 225)))
        painter.drawRoundedRect(rect, radius, radius)

        # Soft ambient gradient reflection inside
        ambient = QLinearGradient(0, 0, float(self.width()), 0)
        if self._state == OverlayState.RECORDING:
            ambient.setColorAt(0.0, QColor(255, 75, 92, 38))
            ambient.setColorAt(0.5, QColor(168, 85, 247, 18))
            ambient.setColorAt(1.0, QColor(56, 189, 248, 30))
        elif self._state == OverlayState.PROCESSING:
            ambient.setColorAt(0.0, QColor(30, 58, 138, 55))
            ambient.setColorAt(1.0, QColor(45, 212, 191, 35))
        elif self._state == OverlayState.ERROR:
            ambient.setColorAt(0.0, QColor(217, 119, 6, 45))
            ambient.setColorAt(1.0, QColor(82, 82, 91, 25))
        else:
            ambient.setColorAt(0.0, QColor(40, 44, 52, 60))
            ambient.setColorAt(1.0, QColor(30, 32, 40, 40))

        painter.setBrush(QBrush(ambient))
        painter.drawRoundedRect(rect, radius, radius)

    def _draw_gradient_border(self, painter: QPainter, rect: QRectF, radius: float) -> None:
        """Draw 1.8px state-dependent gradient border."""
        border_gradient = QLinearGradient(0, 0, float(self.width()), 0)

        if self._state == OverlayState.RECORDING:
            # Coral-red (#FF4B5C) to Sky-blue (#38BDF8)
            border_gradient.setColorAt(0.0, QColor("#FF4B5C"))
            border_gradient.setColorAt(0.35, QColor("#EC4899"))
            border_gradient.setColorAt(0.70, QColor("#818CF8"))
            border_gradient.setColorAt(1.0, QColor("#38BDF8"))
        elif self._state == OverlayState.PROCESSING:
            # Deep blue (#1E3A8A) to Glowing teal/cyan (#2DD4BF)
            border_gradient.setColorAt(0.0, QColor("#1E3A8A"))
            border_gradient.setColorAt(0.5, QColor("#0284C7"))
            border_gradient.setColorAt(1.0, QColor("#2DD4BF"))
        elif self._state == OverlayState.ERROR:
            # Warm amber (#D97706) to Muted gray (#52525B)
            border_gradient.setColorAt(0.0, QColor("#D97706"))
            border_gradient.setColorAt(0.5, QColor("#B45309"))
            border_gradient.setColorAt(1.0, QColor("#52525B"))
        else:
            # Standby subtle border
            border_gradient.setColorAt(0.0, QColor(100, 116, 139, 130))
            border_gradient.setColorAt(1.0, QColor(71, 85, 105, 90))

        border_pen = QPen(QBrush(border_gradient), 1.8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(border_pen)
        painter.drawRoundedRect(rect, radius, radius)

    # -------------------------------------------------------------------------
    # State 1: Recording Visualizer (Match Reference Image Top Pill)
    # -------------------------------------------------------------------------
    def _draw_recording_state(self, painter: QPainter) -> None:
        """Render glowing red dot, '• LIVE' badge, dynamic equalizer + dual sine wave, and 'Esc' keycap."""
        cy = float(self.pill_height) / 2.0

        # 1. Glowing Red Dot with Pulsing Halo
        dot_x = 28.0
        halo = QRadialGradient(dot_x, cy, 14.0)
        halo.setColorAt(0.0, QColor(255, 75, 92, max(20, self._pulse_alpha // 3)))
        halo.setColorAt(0.6, QColor(255, 75, 92, max(10, self._pulse_alpha // 8)))
        halo.setColorAt(1.0, QColor(255, 75, 92, 0))
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(dot_x, cy), 14.0, 14.0)

        # Core solid red dot
        painter.setBrush(QBrush(QColor("#FF4B5C")))
        painter.drawEllipse(QPointF(dot_x, cy), 5.5, 5.5)

        # 2. '• LIVE' Pill Badge
        badge_rect = QRectF(48.0, cy - 9.0, 44.0, 18.0)
        painter.setBrush(QBrush(QColor(36, 40, 50, 200)))
        painter.setPen(QPen(QColor(65, 72, 88, 140), 1.0))
        painter.drawRoundedRect(badge_rect, 9.0, 9.0)

        # Tiny red dot inside badge
        painter.setBrush(QBrush(QColor("#EF4444")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(56.5, cy), 2.2, 2.2)

        # LIVE text
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#F1F5F9")))
        painter.drawText(QRectF(63.0, cy - 9.0, 26.0, 18.0), Qt.AlignmentFlag.AlignCenter, "LIVE")

        # 3. Hybrid Visualizer (Center)
        # Left half: Vertical Equalizer Bars (x = 104 to 154)
        eq_start_x = 104.0
        bar_w = 2.8
        gap = 3.6
        for i in range(self._num_bars):
            h = self._bar_heights[i]
            bx = eq_start_x + i * (bar_w + gap)
            by = cy - (h / 2.0)
            bar_rect = QRectF(bx, by, bar_w, h)

            # Color transitions from coral to violet to sky blue
            t = i / float(self._num_bars - 1)
            bar_color = QColor(
                int(251 * (1 - t) + 56 * t),
                int(113 * (1 - t) + 189 * t),
                int(133 * (1 - t) + 248 * t),
            )
            painter.setBrush(QBrush(bar_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 1.4, 1.4)

        # Right half: Multi-Layered Glowing Sine Waves (x = 168 to 316)
        wave_start_x = 166.0
        wave_end_x = 316.0
        wave_width = wave_end_x - wave_start_x

        # Secondary Translucent Purple Wave (Underneath)
        path_secondary = QPainterPath()
        points_sec: list[QPointF] = []
        for xi in range(int(wave_width) + 1):
            px = wave_start_x + xi
            # Taper envelope to guarantee wave cleanly dampens to 0 at both edges
            envelope = math.sin(xi * math.pi / wave_width)
            # Secondary wave offset phase
            amp = (3.5 + self._smoothed_amplitude * 16.0) * envelope
            py = cy + amp * math.sin((xi * 0.08) + self._wave_phase + 1.2)
            points_sec.append(QPointF(px, py))

        if points_sec:
            path_secondary.moveTo(points_sec[0])
            for pt in points_sec[1:]:
                path_secondary.lineTo(pt)
            pen_sec = QPen(QColor(168, 85, 247, 140), 1.6)
            pen_sec.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen_sec)
            painter.drawPath(path_secondary)

        # Primary Bright Cyan-to-Pink Gradient Wave (On Top)
        path_primary = QPainterPath()
        points_pri: list[QPointF] = []
        for xi in range(int(wave_width) + 1):
            px = wave_start_x + xi
            envelope = math.sin(xi * math.pi / wave_width)
            amp = (5.0 + self._smoothed_amplitude * 22.0) * envelope
            py = cy + amp * math.sin((xi * 0.065) + self._wave_phase)
            points_pri.append(QPointF(px, py))

        if points_pri:
            path_primary.moveTo(points_pri[0])
            for pt in points_pri[1:]:
                path_primary.lineTo(pt)
            wave_grad = QLinearGradient(wave_start_x, 0, wave_end_x, 0)
            wave_grad.setColorAt(0.0, QColor("#F43F5E"))  # Pink/Coral
            wave_grad.setColorAt(0.45, QColor("#A855F7"))  # Purple
            wave_grad.setColorAt(1.0, QColor("#38BDF8"))  # Cyan
            pen_pri = QPen(QBrush(wave_grad), 2.4)
            pen_pri.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen_pri)
            painter.drawPath(path_primary)

    # -------------------------------------------------------------------------
    # State 2: Processing (Match Reference Image Middle Pill)
    # -------------------------------------------------------------------------
    def _draw_processing_state(self, painter: QPainter) -> None:
        """Render glowing cyan/teal flowing harmonic wave orb and 'Processing speech...'."""
        cy = float(self.pill_height) / 2.0
        wave_start_x = 24.0
        wave_width = 118.0

        # Flowing multi-harmonic sine waves (Left)
        # Harmonic C (Deep blue underlayer)
        path_c = QPainterPath()
        for xi in range(int(wave_width) + 1):
            px = wave_start_x + xi
            env = (math.sin(xi * math.pi / wave_width)) ** 1.3
            py = cy + 7.0 * env * math.sin(xi * 0.08 + self._proc_phase + 3.2)
            if xi == 0:
                path_c.moveTo(px, py)
            else:
                path_c.lineTo(px, py)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(59, 130, 246, 120), 1.4))
        painter.drawPath(path_c)

        # Harmonic B (Teal middle layer)
        path_b = QPainterPath()
        for xi in range(int(wave_width) + 1):
            px = wave_start_x + xi
            env = (math.sin(xi * math.pi / wave_width)) ** 1.3
            py = cy + 9.5 * env * math.sin(xi * 0.11 + self._proc_phase + 1.6)
            if xi == 0:
                path_b.moveTo(px, py)
            else:
                path_b.lineTo(px, py)
        painter.setPen(QPen(QColor(6, 182, 212, 170), 1.6))
        painter.drawPath(path_b)

        # Harmonic A (Bright glowing cyan top layer)
        path_a = QPainterPath()
        for xi in range(int(wave_width) + 1):
            px = wave_start_x + xi
            env = (math.sin(xi * math.pi / wave_width)) ** 1.3
            py = cy + 12.0 * env * math.sin(xi * 0.09 + self._proc_phase)
            if xi == 0:
                path_a.moveTo(px, py)
            else:
                path_a.lineTo(px, py)
        pen_a = QPen(QColor("#2DD4BF"), 2.2)
        pen_a.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_a)
        painter.drawPath(path_a)

        # Crisp White Text (Center / Right)
        is_dl = getattr(self, "_is_downloading", False)
        font_size = 9 if (is_dl and len(self._processing_text) > 24) else (10 if len(self._processing_text) > 18 else 11)
        painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#F8FAFC")))
        text_x = 132.0 if is_dl else 150.0
        text_w = 165.0 if is_dl else 180.0
        text_rect = QRectF(text_x, 0.0, text_w, float(self.pill_height))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._processing_text,
        )

        # Slim Glowing Progress Bar along bottom edge when downloading
        if is_dl and self._download_pct is not None:
            pct = max(0.0, min(100.0, float(self._download_pct)))
            bar_x = 24.0
            bar_max_w = float(self.pill_width) - 48.0
            curr_w = max(4.0, bar_max_w * (pct / 100.0))
            bar_y = float(self.pill_height) - 4.5
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(45, 212, 191, 50)))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_max_w, 2.5), 1.25, 1.25)
            painter.setBrush(QBrush(QColor("#38BDF8")))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, curr_w, 2.5), 1.25, 1.25)

    # -------------------------------------------------------------------------
    # State 3: Error (Match Reference Image Bottom Pill)
    # -------------------------------------------------------------------------
    def _draw_error_state(self, painter: QPainter) -> None:
        """Render amber exclamation badge, crisp error message, and subtle refresh icon."""
        cy = float(self.pill_height) / 2.0

        # 1. Circular Amber Exclamation Badge (Left)
        badge_x = 34.0
        badge_r = 13.0
        badge_rect = QRectF(badge_x - badge_r, cy - badge_r, badge_r * 2, badge_r * 2)

        painter.setBrush(QBrush(QColor(217, 119, 6, 45)))
        painter.setPen(QPen(QColor(245, 158, 11, 140), 1.2))
        painter.drawEllipse(badge_rect)

        # Exclamation Mark
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#F59E0B")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "!")

        # 2. Crisp Error Message (Center)
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#FFFFFF")))
        msg_rect = QRectF(60.0, 0.0, 232.0, float(self.pill_height))
        painter.drawText(
            msg_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._error_text or "No speech detected",
        )

        # 3. Subtle Refresh / Retry Icon (Right, before gear button)
        retry_cx = 304.0
        retry_r = 7.5
        retry_rect = QRectF(retry_cx - retry_r, cy - retry_r, retry_r * 2, retry_r * 2)
        pen_retry = QPen(QColor("#94A3B8"), 1.8)
        pen_retry.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen_retry)
        painter.drawArc(retry_rect, int(30 * 16), int(280 * 16))

        # Arrowhead on the arc
        arrow_poly = QPainterPath()
        arrow_poly.moveTo(retry_cx + retry_r - 1.0, cy - 3.5)
        arrow_poly.lineTo(retry_cx + retry_r + 3.0, cy + 1.0)
        arrow_poly.lineTo(retry_cx + retry_r - 4.5, cy + 1.0)
        arrow_poly.closeSubpath()
        painter.setBrush(QBrush(QColor("#94A3B8")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(arrow_poly)

    def _draw_gear_button_at(
        self,
        painter: QPainter,
        cx: float,
        cy: float,
        r: float = 14.0,
        is_matte: bool = False,
        is_fluent: bool = False,
    ) -> None:
        """Render circular settings gear button (⚙) at given center coordinates."""
        if is_matte:
            bg_alpha = 40 if self._hover_gear else 12
            border_alpha = 90 if self._hover_gear else 35
        elif is_fluent:
            bg_alpha = 50 if self._hover_gear else 20
            border_alpha = 100 if self._hover_gear else 40
        else:
            bg_alpha = 40 if self._hover_gear else 18
            border_alpha = 90 if self._hover_gear else 45

        painter.setBrush(QBrush(QColor(255, 255, 255, bg_alpha)))
        painter.setPen(QPen(QColor(255, 255, 255, border_alpha), 1.0))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        icon_color = (
            QColor("#FFFFFF")
            if self._hover_gear
            else (QColor("#A1A1AA") if is_matte else QColor("#94A3B8"))
        )
        painter.setBrush(QBrush(icon_color))
        painter.setPen(Qt.PenStyle.NoPen)

        gear_path = QPainterPath()
        num_teeth = 8
        r_outer = r * 0.58
        r_inner = r * 0.46
        r_hole = r * 0.20

        for i in range(num_teeth):
            angle = i * (2.0 * math.pi / num_teeth)
            a1 = angle - 0.20
            a2 = angle - 0.10
            a3 = angle + 0.10
            a4 = angle + 0.20

            p1 = QPointF(cx + r_inner * math.cos(a1), cy + r_inner * math.sin(a1))
            p2 = QPointF(cx + r_outer * math.cos(a2), cy + r_outer * math.sin(a2))
            p3 = QPointF(cx + r_outer * math.cos(a3), cy + r_outer * math.sin(a3))
            p4 = QPointF(cx + r_inner * math.cos(a4), cy + r_inner * math.sin(a4))

            if i == 0:
                gear_path.moveTo(p1)
            else:
                gear_path.lineTo(p1)
            gear_path.lineTo(p2)
            gear_path.lineTo(p3)
            gear_path.lineTo(p4)

        gear_path.closeSubpath()
        hole_path = QPainterPath()
        hole_path.addEllipse(QPointF(cx, cy), r_hole, r_hole)
        gear_combined = gear_path.subtracted(hole_path)
        painter.drawPath(gear_combined)

    def _draw_gear_button(self, painter: QPainter) -> None:
        """Render circular settings gear button (⚙) on the far right (matches reference image)."""
        self._draw_gear_button_at(painter, 345.0, float(self.pill_height) / 2.0, 15.0)

    def _draw_cancel_download_button(self, painter: QPainter) -> None:
        """Render circular [✕] cancel button when model download is active."""
        if not getattr(self, "_is_downloading", False):
            return

        cx = 308.0
        cy = float(self.pill_height) / 2.0
        r = 12.0

        if getattr(self, "_hover_cancel", False):
            bg_color = QColor(220, 38, 38, 220)
            border_color = QColor(254, 202, 202, 240)
            x_color = QColor("#FFFFFF")
        else:
            bg_color = QColor(185, 28, 28, 90)
            border_color = QColor(239, 68, 68, 160)
            x_color = QColor("#FCA5A5")

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1.2))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        cross_sz = 3.8
        pen_x = QPen(x_color, 2.0)
        pen_x.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_x)
        painter.drawLine(
            QPointF(cx - cross_sz, cy - cross_sz),
            QPointF(cx + cross_sz, cy + cross_sz),
        )
        painter.drawLine(
            QPointF(cx - cross_sz, cy + cross_sz),
            QPointF(cx + cross_sz, cy - cross_sz),
        )

    def _draw_retry_icon(self, painter: QPainter, cx: float, cy: float) -> None:
        """Render subtle refresh/retry circular arrow."""
        r = 7.5
        retry_rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        pen_retry = QPen(QColor("#94A3B8"), 1.8)
        pen_retry.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen_retry)
        painter.drawArc(retry_rect, int(30 * 16), int(280 * 16))

        arrow_poly = QPainterPath()
        arrow_poly.moveTo(cx + r - 1.0, cy - 3.5)
        arrow_poly.lineTo(cx + r + 3.0, cy + 1.0)
        arrow_poly.lineTo(cx + r - 4.5, cy + 1.0)
        arrow_poly.closeSubpath()
        painter.setBrush(QBrush(QColor("#94A3B8")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(arrow_poly)

    def _draw_vector_mic(self, painter: QPainter, cx: float, cy: float) -> None:
        """Render sleek vector microphone glyph."""
        painter.setBrush(QBrush(QColor("#EF4444")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(cx - 3.0, cy - 7.0, 6.0, 10.0), 3.0, 3.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen_arc = QPen(QColor("#EF4444"), 1.6)
        pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_arc)
        painter.drawArc(QRectF(cx - 5.5, cy - 4.0, 11.0, 10.0), int(190 * 16), int(160 * 16))
        painter.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, cy + 9.5))
        painter.drawLine(QPointF(cx - 3.5, cy + 9.5), QPointF(cx + 3.5, cy + 9.5))

    def _draw_warning_triangle(self, painter: QPainter, cx: float, cy: float) -> None:
        """Render amber warning triangle badge."""
        tri = QPainterPath()
        tri.moveTo(cx, cy - 7.0)
        tri.lineTo(cx + 8.0, cy + 7.0)
        tri.lineTo(cx - 8.0, cy + 7.0)
        tri.closeSubpath()
        painter.setBrush(QBrush(QColor(245, 158, 11, 40)))
        painter.setPen(QPen(QColor("#F59E0B"), 1.4))
        painter.drawPath(tri)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#F59E0B")))
        painter.drawText(QRectF(cx - 6.0, cy - 4.0, 12.0, 12.0), Qt.AlignmentFlag.AlignCenter, "!")

    def _draw_wave_orb(self, painter: QPainter, cx: float, cy: float, r: float = 12.5) -> None:
        """Render glowing spherical wave marble / orb matching Image 1 processing state."""
        # Outer soft glow halo
        halo = QRadialGradient(cx, cy, r + 4.0)
        halo.setColorAt(0.0, QColor(34, 211, 238, 70))
        halo.setColorAt(0.7, QColor(34, 211, 238, 25))
        halo.setColorAt(1.0, QColor(34, 211, 238, 0))
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), r + 4.0, r + 4.0)

        # Base sphere
        sphere_path = QPainterPath()
        sphere_path.addEllipse(QPointF(cx, cy), r, r)
        painter.save()
        painter.setClipPath(sphere_path)

        # Deep cyan sphere background
        bg_grad = QLinearGradient(cx, cy - r, cx, cy + r)
        bg_grad.setColorAt(0.0, QColor("#0369A1"))
        bg_grad.setColorAt(1.0, QColor("#082F49"))
        painter.setBrush(QBrush(bg_grad))
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Flowing wave layer inside sphere
        wave_path = QPainterPath()
        wave_path.moveTo(cx - r, cy + r)
        wave_path.lineTo(cx - r, cy)
        steps = 20
        for i in range(steps + 1):
            px = (cx - r) + i * (2.0 * r / steps)
            norm = (px - (cx - r)) / (2.0 * r)
            env = math.sin(norm * math.pi)
            py = cy + 4.0 * env * math.sin(norm * 4.0 + self._proc_phase)
            wave_path.lineTo(px, py)
        wave_path.lineTo(cx + r, cy + r)
        wave_path.closeSubpath()

        wave_grad = QLinearGradient(cx, cy - r, cx, cy + r)
        wave_grad.setColorAt(0.0, QColor("#38BDF8"))
        wave_grad.setColorAt(1.0, QColor("#22D3EE"))
        painter.setBrush(QBrush(wave_grad))
        painter.drawPath(wave_path)

        painter.restore()

        # Crisp glowing rim
        pen_rim = QPen(QColor("#38BDF8"), 1.2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen_rim)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    # -------------------------------------------------------------------------
    # State 4: Standby (Subtle Minimal Ghost State)
    # -------------------------------------------------------------------------
    def _draw_standby_state(self, painter: QPainter) -> None:
        """Render minimal ghost standby state with muted indicator."""
        cy = float(self.pill_height) / 2.0

        # Subtle standby dot
        painter.setBrush(QBrush(QColor("#64748B")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(28.0, cy), 4.0, 4.0)

        # Subtle label
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QPen(QColor("#94A3B8")))
        painter.drawText(
            QRectF(42.0, 0.0, 200.0, float(self.pill_height)),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "VoiceTyper Standby",
        )

    # -------------------------------------------------------------------------
    # Theme 2: Neon Glass Pill (Matches Image 1 Reference)
    # -------------------------------------------------------------------------
    def _paint_neon_glass(self, painter: QPainter, rect: QRectF) -> None:
        """Render Theme 2: Neon Glass Pill matching reference design with divider and gear."""
        radius = 28.0
        cy = float(self.pill_height) / 2.0

        # 1. Frosted obsidian glass capsule background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(15, 17, 23, 235)))
        painter.drawRoundedRect(rect, radius, radius)

        # 2. Ambient inner gradient glow
        ambient = QLinearGradient(0, 0, float(self.width()), 0)
        if self._state == OverlayState.RECORDING:
            ambient.setColorAt(0.0, QColor(255, 75, 92, 38))
            ambient.setColorAt(0.5, QColor(168, 85, 247, 18))
            ambient.setColorAt(1.0, QColor(56, 189, 248, 30))
        elif self._state == OverlayState.PROCESSING:
            ambient.setColorAt(0.0, QColor(30, 58, 138, 55))
            ambient.setColorAt(1.0, QColor(45, 212, 191, 35))
        elif self._state == OverlayState.ERROR:
            ambient.setColorAt(0.0, QColor(217, 119, 6, 45))
            ambient.setColorAt(1.0, QColor(63, 63, 70, 25))
        else:
            ambient.setColorAt(0.0, QColor(30, 41, 59, 35))
            ambient.setColorAt(1.0, QColor(15, 23, 42, 25))

        painter.setBrush(QBrush(ambient))
        painter.drawRoundedRect(rect, radius, radius)

        # 3. State content
        if self._state == OverlayState.RECORDING:
            # Glowing red dot with pulsing halo (Matching Image 1: no LIVE badge)
            dot_x = 28.0
            halo = QRadialGradient(dot_x, cy, 14.0)
            halo.setColorAt(0.0, QColor(255, 75, 92, max(20, self._pulse_alpha // 3)))
            halo.setColorAt(0.6, QColor(255, 75, 92, max(10, self._pulse_alpha // 8)))
            halo.setColorAt(1.0, QColor(255, 75, 92, 0))
            painter.setBrush(QBrush(halo))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(dot_x, cy), 14.0, 14.0)
            painter.setBrush(QBrush(QColor("#FF4B5C")))
            painter.drawEllipse(QPointF(dot_x, cy), 5.5, 5.5)

            # Central Voice-Reactive Wave with nestled micro-equalizer bars (x = 52 to 272)
            wave_start_x = 52.0
            wave_end_x = 272.0
            wave_w = wave_end_x - wave_start_x

            # Micro-equalizer needles in center under the wave crest (x = 110 to 190)
            eq_start_x = 112.0
            bar_w = 2.0
            gap = 4.0
            num_eq = 14
            for i in range(num_eq):
                phase_i = i * 0.45 + self._wave_phase * 1.5
                osc = (math.sin(phase_i) + 1.0) / 2.0
                h = 3.0 + 16.0 * self._smoothed_amplitude * (0.3 + 0.7 * osc)
                bx = eq_start_x + i * (bar_w + gap)
                by = cy - (h / 2.0)
                # Cyan-tinted equalizer lines
                painter.setBrush(QBrush(QColor(56, 189, 248, int(140 + 115 * osc))))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QRectF(bx, by, bar_w, h), 1.0, 1.0)

            # Secondary translucent purple wave
            path_sec = QPainterPath()
            for xi in range(int(wave_w) + 1):
                px = wave_start_x + xi
                env = math.sin(xi * math.pi / wave_w)
                amp = (3.5 + self._smoothed_amplitude * 14.0) * env
                py = cy + amp * math.sin((xi * 0.08) + self._wave_phase + 1.2)
                if xi == 0:
                    path_sec.moveTo(px, py)
                else:
                    path_sec.lineTo(px, py)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(168, 85, 247, 130), 1.6))
            painter.drawPath(path_sec)

            # Primary luminous cyan wave on top
            path_pri = QPainterPath()
            for xi in range(int(wave_w) + 1):
                px = wave_start_x + xi
                env = math.sin(xi * math.pi / wave_w)
                amp = (5.0 + self._smoothed_amplitude * 18.0) * env
                py = cy + amp * math.sin((xi * 0.065) + self._wave_phase)
                if xi == 0:
                    path_pri.moveTo(px, py)
                else:
                    path_pri.lineTo(px, py)
            w_grad = QLinearGradient(wave_start_x, 0, wave_end_x, 0)
            w_grad.setColorAt(0.0, QColor("#38BDF8"))
            w_grad.setColorAt(0.5, QColor("#60A5FA"))
            w_grad.setColorAt(1.0, QColor("#818CF8"))
            pen_pri = QPen(QBrush(w_grad), 2.2)
            pen_pri.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_pri)
            painter.drawPath(path_pri)

        elif self._state == OverlayState.PROCESSING:
            # Glowing circular wave orb / marble (Matching Image 1 Processing)
            self._draw_wave_orb(painter, 30.0, cy, 12.5)

            is_dl = getattr(self, "_is_downloading", False)
            font_size = 9 if (is_dl and len(self._processing_text) > 24) else (10 if len(self._processing_text) > 18 else 11)
            painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#F8FAFC")))
            text_w = 230.0 if is_dl else 260.0
            painter.drawText(
                QRectF(56.0, 0.0, text_w, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._processing_text,
            )

            if is_dl and self._download_pct is not None:
                pct = max(0.0, min(100.0, float(self._download_pct)))
                bar_x = 20.0
                bar_max_w = float(self.pill_width) - 40.0
                curr_w = max(4.0, bar_max_w * (pct / 100.0))
                bar_y = float(self.pill_height) - 4.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(56, 189, 248, 50)))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_max_w, 2.0), 1.0, 1.0)
                painter.setBrush(QBrush(QColor("#38BDF8")))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, curr_w, 2.0), 1.0, 1.0)

        elif self._state == OverlayState.ERROR:
            # Amber ! badge
            badge_rect = QRectF(17.0, cy - 13.0, 26.0, 26.0)
            painter.setBrush(QBrush(QColor(217, 119, 6, 45)))
            painter.setPen(QPen(QColor(245, 158, 11, 140), 1.2))
            painter.drawEllipse(badge_rect)
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#F59E0B")))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "!")

            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.drawText(
                QRectF(56.0, 0.0, 230.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._error_text or "No speech detected",
            )
            self._draw_retry_icon(painter, 296.0, cy)

        elif self._state == OverlayState.STANDBY:
            painter.setBrush(QBrush(QColor("#64748B")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(28.0, cy), 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#94A3B8")))
            painter.drawText(
                QRectF(42.0, 0.0, 200.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "VoiceTyper Standby",
            )

        # 4. Vertical Hairline Divider at x = 324.0
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1.0))
        painter.drawLine(QPointF(324.0, 14.0), QPointF(324.0, 42.0))

        # 5. Gear button centered at cx = 350.0 & Cancel button
        self._draw_gear_button_at(painter, 350.0, cy, 14.0)
        self._draw_cancel_download_button(painter)

        # 6. Thin 1.0px Gradient Border
        b_grad = QLinearGradient(0, 0, float(self.width()), 0)
        if self._state == OverlayState.RECORDING:
            b_grad.setColorAt(0.0, QColor("#FF4B5C"))
            b_grad.setColorAt(0.35, QColor("#EC4899"))
            b_grad.setColorAt(0.70, QColor("#818CF8"))
            b_grad.setColorAt(1.0, QColor("#38BDF8"))
        elif self._state == OverlayState.PROCESSING:
            b_grad.setColorAt(0.0, QColor("#1E3A8A"))
            b_grad.setColorAt(0.5, QColor("#0284C7"))
            b_grad.setColorAt(1.0, QColor("#2DD4BF"))
        elif self._state == OverlayState.ERROR:
            b_grad.setColorAt(0.0, QColor("#D97706"))
            b_grad.setColorAt(0.5, QColor("#B45309"))
            b_grad.setColorAt(1.0, QColor("#3F3F46"))
        else:
            b_grad.setColorAt(0.0, QColor(51, 65, 85, 140))
            b_grad.setColorAt(1.0, QColor(30, 41, 59, 100))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QBrush(b_grad), 1.0))
        painter.drawRoundedRect(rect, radius, radius)

    # -------------------------------------------------------------------------
    # Theme 3: Matte Dynamic Island (Matches Image 2 Reference)
    # -------------------------------------------------------------------------
    def _paint_matte_island(self, painter: QPainter, rect: QRectF) -> None:
        """Render Theme 3: Matte Dynamic Island matching reference design."""
        radius = 28.0
        cy = float(self.pill_height) / 2.0

        # 1. Ultra-clean matte carbon black capsule (#121214)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(18, 18, 20, 252)))
        painter.drawRoundedRect(rect, radius, radius)

        # 2. State content
        if self._state == OverlayState.RECORDING:
            # Crimson mic icon
            mic_cx = 28.0
            painter.setBrush(QBrush(QColor(30, 30, 34)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(mic_cx, cy), 13.0, 13.0)
            self._draw_vector_mic(painter, mic_cx, cy)

            # Fine white/silver vertical spectrum bars (x = 52 to 148)
            spec_start_x = 52.0
            bar_w = 2.0
            gap = 5.0
            num_spec_bars = 14
            for i in range(num_spec_bars):
                phase_i = i * 0.40 + self._wave_phase * 1.5
                osc = (math.sin(phase_i) + 1.0) / 2.0
                h = 4.0 + 22.0 * self._smoothed_amplitude * (0.3 + 0.7 * osc)
                bx = spec_start_x + i * (bar_w + gap)
                by = cy - (h / 2.0)
                val = int(220 + 35 * osc)
                painter.setBrush(QBrush(QColor(val, val, val)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QRectF(bx, by, bar_w, h), 1.0, 1.0)

            # Delicate silver sine wave (x = 158 to 274)
            wave_start_x = 158.0
            wave_end_x = 274.0
            wave_w = wave_end_x - wave_start_x
            path_w = QPainterPath()
            for xi in range(int(wave_w) + 1):
                px = wave_start_x + xi
                env = math.sin(xi * math.pi / wave_w)
                amp = (4.0 + self._smoothed_amplitude * 16.0) * env
                py = cy + amp * math.sin((xi * 0.08) + self._wave_phase)
                if xi == 0:
                    path_w.moveTo(px, py)
                else:
                    path_w.lineTo(px, py)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen_w = QPen(QColor(161, 161, 170, 200), 1.8)
            pen_w.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_w)
            painter.drawPath(path_w)

        elif self._state == OverlayState.PROCESSING:
            # 3 animated pulsing white/silver dots (Dynamic Island style)
            dot_xs = [38.0, 52.0, 66.0]
            for i, dx in enumerate(dot_xs):
                pulse = (math.sin(self._proc_phase * 2.2 + i * 1.0) + 1.0) / 2.0
                r = 3.0 + 1.8 * pulse
                alpha = int(100 + 155 * pulse)
                painter.setBrush(QBrush(QColor(244, 244, 245, alpha)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(dx, cy), r, r)

            is_dl = getattr(self, "_is_downloading", False)
            font_size = 9 if (is_dl and len(self._processing_text) > 24) else (10 if len(self._processing_text) > 18 else 11)
            painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#F4F4F5")))
            text_w = 210.0 if is_dl else 230.0
            painter.drawText(
                QRectF(84.0, 0.0, text_w, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._processing_text,
            )

            if is_dl and self._download_pct is not None:
                pct = max(0.0, min(100.0, float(self._download_pct)))
                bar_x = 20.0
                bar_max_w = float(self.pill_width) - 40.0
                curr_w = max(4.0, bar_max_w * (pct / 100.0))
                bar_y = float(self.pill_height) - 4.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(244, 244, 245, 40)))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_max_w, 2.0), 1.0, 1.0)
                painter.setBrush(QBrush(QColor("#F4F4F5")))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, curr_w, 2.0), 1.0, 1.0)

        elif self._state == OverlayState.ERROR:
            # Soft red dot with subtle halo
            dot_cx = 30.0
            painter.setBrush(QBrush(QColor(239, 68, 68, 45)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(dot_cx, cy), 11.0, 11.0)
            painter.setBrush(QBrush(QColor("#EF4444")))
            painter.drawEllipse(QPointF(dot_cx, cy), 4.5, 4.5)

            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#F4F4F5")))
            painter.drawText(
                QRectF(52.0, 0.0, 236.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._error_text or "No speech detected",
            )
            self._draw_retry_icon(painter, 296.0, cy)

        elif self._state == OverlayState.STANDBY:
            painter.setBrush(QBrush(QColor("#71717A")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(28.0, cy), 3.5, 3.5)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#A1A1AA")))
            painter.drawText(
                QRectF(42.0, 0.0, 200.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "VoiceTyper Ready",
            )

        # 3. Vertical Hairline Divider at x = 324.0
        painter.setPen(QPen(QColor(42, 42, 46, 220), 1.0))
        painter.drawLine(QPointF(324.0, 15.0), QPointF(324.0, 41.0))

        # 4. Settings gear button at cx = 350.0 & Cancel button
        self._draw_gear_button_at(painter, 350.0, cy, 14.0, is_matte=True)
        self._draw_cancel_download_button(painter)

        # 5. Crisp 1.0px hairline border in #2A2A2E
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#2A2A2E"), 1.0))
        painter.drawRoundedRect(rect, radius, radius)

    # -------------------------------------------------------------------------
    # Theme 4: Fluent Mica Card (Matches Image 3 Reference)
    # -------------------------------------------------------------------------
    def _paint_fluent_mica(self, painter: QPainter, rect: QRectF) -> None:
        """Render Theme 4: Fluent Mica Card (Windows 11 card with 16px corner radius)."""
        radius = 16.0
        cy = float(self.pill_height) / 2.0

        # 1. Smoked translucent glass (#18181B)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(24, 24, 27, 230)))
        painter.drawRoundedRect(rect, radius, radius)

        # 2. State content
        if self._state == OverlayState.RECORDING:
            # Coral-red dot in circular tint
            dot_cx = 30.0
            painter.setBrush(QBrush(QColor(239, 68, 68, 30)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(dot_cx, cy), 13.0, 13.0)
            painter.setBrush(QBrush(QColor("#EF4444")))
            painter.drawEllipse(QPointF(dot_cx, cy), 5.0, 5.0)

            # Fluid blue-to-violet gradient sine wave (x = 56 to 274)
            wave_start_x = 56.0
            wave_end_x = 274.0
            wave_w = wave_end_x - wave_start_x

            # Wave underlayer
            path_sub = QPainterPath()
            for xi in range(int(wave_w) + 1):
                px = wave_start_x + xi
                env = math.sin(xi * math.pi / wave_w)
                amp = (3.5 + self._smoothed_amplitude * 14.0) * env
                py = cy + amp * math.sin((xi * 0.075) + self._wave_phase + 1.4)
                if xi == 0:
                    path_sub.moveTo(px, py)
                else:
                    path_sub.lineTo(px, py)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(139, 92, 246, 120), 1.5))
            painter.drawPath(path_sub)

            # Wave main layer
            path_main = QPainterPath()
            for xi in range(int(wave_w) + 1):
                px = wave_start_x + xi
                env = math.sin(xi * math.pi / wave_w)
                amp = (5.0 + self._smoothed_amplitude * 20.0) * env
                py = cy + amp * math.sin((xi * 0.065) + self._wave_phase)
                if xi == 0:
                    path_main.moveTo(px, py)
                else:
                    path_main.lineTo(px, py)
            w_grad = QLinearGradient(wave_start_x, 0, wave_end_x, 0)
            w_grad.setColorAt(0.0, QColor("#60A5FA"))
            w_grad.setColorAt(0.5, QColor("#8B5CF6"))
            w_grad.setColorAt(1.0, QColor("#EC4899"))
            pen_m = QPen(QBrush(w_grad), 2.2)
            pen_m.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_m)
            painter.drawPath(path_main)

        elif self._state == OverlayState.PROCESSING:
            # Circular gradient progress ring (cx = 32.0, cy = 28.0)
            ring_cx = 32.0
            ring_r = 10.0
            ring_rect = QRectF(ring_cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)

            # Background subtle ring
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 20), 2.0))
            painter.drawEllipse(QPointF(ring_cx, cy), ring_r, ring_r)

            # Spinning gradient arc
            start_angle = int((self._proc_phase * 180) % 360 * 16)
            arc_grad = QLinearGradient(ring_cx - ring_r, cy - ring_r, ring_cx + ring_r, cy + ring_r)
            arc_grad.setColorAt(0.0, QColor("#38BDF8"))
            arc_grad.setColorAt(1.0, QColor("#A855F7"))
            pen_ring = QPen(QBrush(arc_grad), 2.2)
            pen_ring.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_ring)
            painter.drawArc(ring_rect, start_angle, int(260 * 16))

            is_dl = getattr(self, "_is_downloading", False)
            font_size = 9 if (is_dl and len(self._processing_text) > 24) else (10 if len(self._processing_text) > 18 else 11)
            painter.setFont(QFont("Segoe UI", font_size, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#F8FAFC")))
            text_w = 230.0 if is_dl else 260.0
            painter.drawText(
                QRectF(56.0, 0.0, text_w, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._processing_text,
            )

            if is_dl and self._download_pct is not None:
                pct = max(0.0, min(100.0, float(self._download_pct)))
                bar_x = 16.0
                bar_max_w = float(self.pill_width) - 32.0
                curr_w = max(4.0, bar_max_w * (pct / 100.0))
                bar_y = float(self.pill_height) - 4.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(96, 165, 250, 40)))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_max_w, 2.0), 1.0, 1.0)
                painter.setBrush(QBrush(QColor("#38BDF8")))
                painter.drawRoundedRect(QRectF(bar_x, bar_y, curr_w, 2.0), 1.0, 1.0)

        elif self._state == OverlayState.ERROR:
            # Warning triangle in soft circle
            self._draw_warning_triangle(painter, 30.0, cy)
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#F1F5F9")))
            painter.drawText(
                QRectF(54.0, 0.0, 234.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._error_text or "No speech detected",
            )
            self._draw_retry_icon(painter, 296.0, cy)

        elif self._state == OverlayState.STANDBY:
            painter.setBrush(QBrush(QColor("#60A5FA")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(28.0, cy), 3.5, 3.5)
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.setPen(QPen(QColor("#94A3B8")))
            painter.drawText(
                QRectF(42.0, 0.0, 200.0, float(self.pill_height)),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                "VoiceTyper Standby",
            )

        # 3. Vertical Hairline Divider at x = 324.0
        painter.setPen(QPen(QColor(255, 255, 255, 28), 1.0))
        painter.drawLine(QPointF(324.0, 14.0), QPointF(324.0, 42.0))

        # 4. Settings gear button at cx = 350.0 & Cancel button
        self._draw_gear_button_at(painter, 350.0, cy, 14.0, is_fluent=True)
        self._draw_cancel_download_button(painter)

        # 5. Top subtle highlight edge
        painter.setPen(QPen(QColor(255, 255, 255, 55), 1.0))
        painter.drawLine(QPointF(16.0, 1.5), QPointF(float(self.width()) - 16.0, 1.5))

        # 6. Card outline border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0))
        painter.drawRoundedRect(rect, radius, radius)


class OverlayPill(QWidget, BaseOverlayRenderer):
    """Floating, draggable, non-activating status pill with voice-reactive visuals."""

    position_changed = pyqtSignal(int, int)  # type: ignore[call-arg]  # (x, y) coordinates on drag release
    settings_clicked = pyqtSignal()  # type: ignore[call-arg]  # Emitted when settings gear icon is clicked
    cancel_download_clicked = pyqtSignal()  # type: ignore[call-arg]  # Emitted when [✕] download cancel button is clicked

    def __init__(
        self,
        audio_recorder: Any = None,
        config: Any = None,
        config_manager: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.audio_recorder = audio_recorder
        self.config = config
        self.config_manager = config_manager

        # Non-activating, always-on-top, frameless tool window
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setWindowFlags(flags)

        # Do NOT set WindowTransparentForInput: click & drag must work!
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        # Fixed pill dimensions matching reference design
        self.pill_width = 380
        self.pill_height = 56
        self.setFixedSize(self.pill_width, self.pill_height)

        # State management
        self._state = OverlayState.HIDDEN
        self._error_text = ""
        self._processing_text = "Processing speech..."

        # Model download state tracking
        self._is_downloading = False
        self._download_pct: float | None = None
        self._download_text = ""
        self._hover_cancel = False
        self._cancel_pressed = False

        # Drag-and-drop & button interaction tracking
        self._is_dragging = False
        self._drag_start_pos = QPoint()
        self._hover_gear = False
        self._gear_pressed = False

        # Visual theme: "classic", "neon_glass", "matte_island", "fluent_mica"
        self.theme: str = getattr(self.config, "overlay_theme", "classic") if self.config else "classic"

        # Audio amplitude smoothing (RMS linear interpolation 0.25 - 0.3)
        self._raw_amplitude = 0.0
        self._smoothed_amplitude = 0.0

        # Animation parameters
        self._wave_phase = 0.0
        self._proc_phase = 0.0
        self._pulse_alpha = 255
        self._pulse_dir = -4
        self._num_bars = 9
        self._bar_phases = [i * 0.45 for i in range(self._num_bars)]
        self._bar_heights = [4.0] * self._num_bars

        # 60 FPS animation timer (~16 ms) with precise timing
        self._anim_timer = QTimer(self)
        self._anim_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_animation_tick)

        # 60-Second Inactivity Timer for Standby state auto-hide
        self._inactivity_timer = QTimer(self)
        self._inactivity_timer.setInterval(60000)  # 60 seconds
        self._inactivity_timer.setSingleShot(True)
        self._inactivity_timer.timeout.connect(self._on_inactivity_timeout)

        # Error auto-revert timer
        self._error_timer = QTimer(self)
        self._error_timer.setInterval(1800)  # 1.8 seconds
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self.show_standby)

        # Fade animations (QPropertyAnimation on native windowOpacity)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setWindowOpacity(1.0)

        # Initial positioning
        self.reposition(force_default=False)

    def _apply_win32_styles(self) -> None:
        """Apply Win32 WS_EX_NOACTIVATE to guarantee zero keyboard focus stealing."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.winId())
            current_style = GetWindowLongPtr(hwnd, GWL_EXSTYLE)
            # Include NOACTIVATE, TOOLWINDOW, TOPMOST. Do NOT include TRANSPARENT.
            new_style = (current_style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~0x00000020
            if new_style != current_style:
                SetWindowLongPtr(hwnd, GWL_EXSTYLE, new_style)
        except Exception:
            pass

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._apply_win32_styles()

    # -------------------------------------------------------------------------
    # Positioning & Multi-Monitor Support
    # -------------------------------------------------------------------------
    def _is_valid_on_any_screen(self, x: int, y: int) -> bool:
        """Check if coordinates intersect with any currently connected screen."""
        pill_rect = QRect(x, y, self.pill_width, self.pill_height)
        for screen in QGuiApplication.screens():
            if screen.geometry().intersects(pill_rect):
                return True
        return False

    def reposition(self, force_default: bool = False) -> None:
        """Position pill at saved config coordinates or default above taskbar clock."""
        saved_pos = getattr(self.config, "overlay_pos", None) if self.config else None

        if not force_default and saved_pos and len(saved_pos) == 2:
            x, y = int(saved_pos[0]), int(saved_pos[1])
            if self._is_valid_on_any_screen(x, y):
                self.move(x, y)
                return

        # Default position: bottom-right of primary screen, above taskbar clock (~12px margin)
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            default_x = avail.x() + avail.width() - self.pill_width - 12
            default_y = avail.y() + avail.height() - self.pill_height - 12
            self.move(default_x, default_y)

    def set_theme(self, theme: str) -> None:
        """Switch visual theme live and update overlay."""
        valid_themes = {"classic", "neon_glass", "matte_island", "fluent_mica"}
        if theme not in valid_themes:
            logger.warning("Unknown theme '%s', falling back to 'classic'", theme)
            theme = "classic"
        self.theme = theme
        if self.config:
            self.config.overlay_theme = theme
        if self.config_manager and self.config:
            try:
                self.config_manager.save_config(self.config)
            except Exception as err:
                logger.debug("Failed to auto-save theme change: %s", err)
        self.update()

    def reset_position(self) -> None:
        """Reset pill to the default position above the taskbar clock."""
        if self.config_manager:
            self.config_manager.reset_overlay_position()
        if self.config:
            self.config.overlay_pos = None
        self.reposition(force_default=True)

    # -------------------------------------------------------------------------
    # Drag-to-Move & Gear Button Click Event Handlers
    # -------------------------------------------------------------------------
    def _is_in_gear_button(self, pos: QPointF | QPoint) -> bool:
        """Check if position is inside the circular gear button."""
        if self._is_downloading and self._is_in_cancel_button(pos):
            return False
        if pos.x() >= 324.0:
            return True
        cx = 345.0
        cy = float(self.pill_height) / 2.0
        dx = pos.x() - cx
        dy = pos.y() - cy
        return (dx * dx + dy * dy) <= (18.0 * 18.0)

    def _is_in_cancel_button(self, pos: QPointF | QPoint) -> bool:
        """Check if position is inside the circular [✕] cancel button."""
        if not self._is_downloading:
            return False
        cx = 308.0
        cy = float(self.pill_height) / 2.0
        dx = pos.x() - cx
        dy = pos.y() - cy
        return (dx * dx + dy * dy) <= (14.0 * 14.0)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self._is_in_cancel_button(pos):
                self._cancel_pressed = True
                self._gear_pressed = False
                self._is_dragging = False
                event.accept()
                return

            if self._is_in_gear_button(pos):
                self._gear_pressed = True
                self._cancel_pressed = False
                self._is_dragging = False
                event.accept()
                return

            self._cancel_pressed = False
            self._gear_pressed = False
            self._is_dragging = True
            self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if not event:
            return

        if self._is_dragging:
            new_pos = event.globalPosition().toPoint() - self._drag_start_pos
            self.move(new_pos)
            event.accept()
            return

        # Check hover over cancel button
        in_cancel = self._is_in_cancel_button(event.position())
        if in_cancel != self._hover_cancel:
            self._hover_cancel = in_cancel
            if in_cancel:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif not self._hover_gear:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

        # Check hover over gear button
        in_gear = self._is_in_gear_button(event.position())
        if in_gear != self._hover_gear:
            self._hover_gear = in_gear
            if in_gear:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif not self._hover_cancel:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event and event.button() == Qt.MouseButton.LeftButton:
            if self._cancel_pressed:
                self._cancel_pressed = False
                if self._is_in_cancel_button(event.position()):
                    logger.debug("Overlay cancel download button clicked; emitting cancel_download_clicked")
                    self.cancel_download_clicked.emit()
                event.accept()
                return

            if self._gear_pressed:
                self._gear_pressed = False
                if self._is_in_gear_button(event.position()):
                    logger.debug("Overlay gear icon clicked; emitting settings_clicked")
                    self.settings_clicked.emit()
                event.accept()
                return

            if self._is_dragging:
                self._is_dragging = False
                x = self.x()
                y = self.y()
                self.position_changed.emit(x, y)
                if self.config:
                    self.config.overlay_pos = [x, y]
                if self.config_manager:
                    self.config_manager.save_overlay_position(x, y)
                event.accept()

    def leaveEvent(self, event: Any) -> None:
        changed = False
        if self._hover_gear:
            self._hover_gear = False
            changed = True
        if self._hover_cancel:
            self._hover_cancel = False
            changed = True
        if changed:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event and event.key() == Qt.Key.Key_Escape:
            if self._is_downloading:
                logger.info("Esc pressed on overlay; cancelling model download")
                self.cancel_download_clicked.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    # -------------------------------------------------------------------------
    # State Transitions & Smart Opacity
    # -------------------------------------------------------------------------
    def set_amplitude(self, rms: float) -> None:
        """Thread-safe update of current audio amplitude."""
        self._raw_amplitude = max(0.0, float(rms))

    @pyqtSlot(str, str)
    def set_state(self, state: OverlayState | str, message: str = "") -> None:
        """Slot to transition overlay state."""
        state_str = state.value if isinstance(state, OverlayState) else str(state).upper()
        if state_str == OverlayState.RECORDING.value:
            self.show_recording()
        elif state_str == OverlayState.PROCESSING.value:
            self.show_processing()
        elif state_str == OverlayState.ERROR.value:
            self.show_error(message or "Error")
        elif state_str == OverlayState.STANDBY.value:
            self.show_standby()
        else:
            self.hide_overlay()

    def _fade_to(self, target_opacity: float, duration_ms: int = 250, on_finished: Any = None) -> None:
        """Smoothly animate window opacity using QPropertyAnimation."""
        self._opacity_anim.stop()
        try:
            self._opacity_anim.finished.disconnect()
        except Exception:
            pass

        self._opacity_anim.setDuration(duration_ms)
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(max(0.0, min(1.0, float(target_opacity))))
        if on_finished:
            self._opacity_anim.finished.connect(on_finished)
        self._opacity_anim.start()

    def show_recording(self) -> None:
        """Activate Recording state at full opacity with live waveform visualizer."""
        self._inactivity_timer.stop()
        self._error_timer.stop()
        self._state = OverlayState.RECORDING

        self._fade_to(1.0, duration_ms=120)
        self.show()
        self.raise_()
        self._apply_win32_styles()

        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_processing(self, message: str = "Processing speech...") -> None:
        """Activate Processing state with flowing harmonic wave orb."""
        self._inactivity_timer.stop()
        self._error_timer.stop()
        self._state = OverlayState.PROCESSING
        self._processing_text = message or "Processing speech..."
        if "download" in self._processing_text.lower() or "تحميل" in self._processing_text:
            self._is_downloading = True

        self._fade_to(1.0, duration_ms=120)
        self.show()
        self.raise_()
        self._apply_win32_styles()

        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def show_download(self, text: str = "Downloading model...", pct: float = 0.0, model_name: str = "") -> None:
        """Activate real-time download display with progress bar, numbers, and cancel button."""
        self._is_downloading = True
        self._download_pct = pct
        self._processing_text = text
        self._inactivity_timer.stop()
        self._error_timer.stop()
        self._state = OverlayState.PROCESSING

        self._fade_to(1.0, duration_ms=120)
        self.show()
        self.raise_()
        self._apply_win32_styles()

        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    def update_download_progress(self, text: str, pct: float, model_name: str = "") -> None:
        """Update live download progress numbers and progress bar."""
        self._is_downloading = True
        self._download_pct = pct
        self._processing_text = text
        if self._state != OverlayState.PROCESSING:
            self._state = OverlayState.PROCESSING
            self.setWindowOpacity(1.0)
            self.show()
        self.update()

    def finish_download(self, success: bool, message: str = "") -> None:
        """Finish download: display checkmark/ready message on success or error on failure."""
        self._is_downloading = False
        self._download_pct = None
        self._hover_cancel = False
        self._cancel_pressed = False
        if success:
            lang = getattr(self.config, "ui_language", "ar") if self.config else "ar"
            ready_msg = "الموديل جاهز للاستخدام ✓" if lang == "ar" else "Model ready ✓"
            self._processing_text = message or ready_msg
            self._state = OverlayState.PROCESSING
            self.setWindowOpacity(1.0)
            self.update()
            # Fade to standby after 2.5 seconds
            QTimer.singleShot(2500, self._on_download_complete_timeout)
        else:
            self.show_error(message or "Download cancelled")

    def _on_download_complete_timeout(self) -> None:
        if not self._is_downloading and self._state == OverlayState.PROCESSING:
            self.show_standby()

    def cancel_active_download(self) -> None:
        """Reset download state cleanly."""
        self._is_downloading = False
        self._download_pct = None
        self._hover_cancel = False
        self._cancel_pressed = False
        self.show_standby()

    def show_error(self, message: str) -> None:
        """Display error badge and toast message for 1.8 seconds."""
        self._inactivity_timer.stop()
        self._state = OverlayState.ERROR
        self._error_text = message

        self._fade_to(1.0, duration_ms=120)
        self.show()
        self.raise_()
        self._apply_win32_styles()
        self.update()

        self._error_timer.start()

    def show_standby(self) -> None:
        """Smoothly fade down to subtle ghost state (0.35 opacity) and start 60s timeout."""
        if getattr(self, "_is_downloading", False):
            return
        self._state = OverlayState.STANDBY
        self._anim_timer.stop()  # Stop 60 FPS timer to save 100% CPU

        self._fade_to(0.35, duration_ms=300)
        self.show()
        self.raise_()
        self._apply_win32_styles()
        self._inactivity_timer.start()
        self.update()

    def _on_inactivity_timeout(self) -> None:
        """Fade out completely and hide when inactive for 60 seconds."""
        def complete_hide() -> None:
            self._state = OverlayState.HIDDEN
            self.hide()

        self._fade_to(0.0, duration_ms=400, on_finished=complete_hide)

    def hide_overlay(self) -> None:
        """Immediately hide overlay."""
        self._inactivity_timer.stop()
        self._error_timer.stop()
        self._anim_timer.stop()
        self._state = OverlayState.HIDDEN
        self.hide()

    # -------------------------------------------------------------------------
    # 60 FPS Visualizer Engine
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Tick 60 FPS visualizer variables."""
        if self._state == OverlayState.RECORDING:
            # Poll RMS from audio recorder if provided
            if self.audio_recorder:
                self._raw_amplitude = self.audio_recorder.get_latest_amplitude()

            # Linear interpolation smoothing (0.28 factor)
            live_amp = min(1.0, self._raw_amplitude * 9.0)
            self._smoothed_amplitude = self._smoothed_amplitude * 0.72 + live_amp * 0.28

            # Update wave phase
            self._wave_phase += 0.08

            # Animate equalizer bar heights
            base_h = 3.5
            max_h = 26.0
            for i in range(self._num_bars):
                self._bar_phases[i] += 0.12
                osc = (math.sin(self._bar_phases[i]) + 1.0) / 2.0
                target = base_h + (max_h - base_h) * self._smoothed_amplitude * (0.35 + 0.65 * osc)
                self._bar_heights[i] = self._bar_heights[i] * 0.70 + target * 0.30

            # Pulse red recording dot
            self._pulse_alpha += self._pulse_dir * 5
            if self._pulse_alpha <= 110:
                self._pulse_alpha = 110
                self._pulse_dir = 1
            elif self._pulse_alpha >= 255:
                self._pulse_alpha = 255
                self._pulse_dir = -1

            self.update()

        elif self._state == OverlayState.PROCESSING:
            self._proc_phase += 0.08
            self.update()

    def preview_state(self, state: OverlayState, message: str = "") -> None:
        """Temporarily display a state at 1.0 opacity for live theme inspection in Settings."""
        self._inactivity_timer.stop()
        self._error_timer.stop()
        self._state = state
        if state == OverlayState.ERROR:
            self._error_text = message or "No speech detected"
        elif state == OverlayState.PROCESSING:
            self._processing_text = message or "Processing speech..."

        self._fade_to(1.0, duration_ms=100)
        self.show()
        self.raise_()
        self._apply_win32_styles()

        if not self._anim_timer.isActive():
            self._anim_timer.start()
        self.update()

    # -------------------------------------------------------------------------
    # Painting & Rendering (Matching Reference Design)
    # -------------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent | None) -> None:
        if self._state == OverlayState.HIDDEN:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        rect = QRectF(1.0, 1.0, float(self.width() - 2), float(self.height() - 2))

        self.paint_theme_overlay(painter, rect)

        painter.end()





class ThemePreviewWidget(QWidget, BaseOverlayRenderer):
    """Interactive 60 FPS animated preview widget embedded in SettingsDialog."""

    def __init__(self, theme: str = "classic", parent: QWidget | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.theme = theme
        self.pill_width = 380
        self.pill_height = 56
        self.setFixedSize(self.pill_width, self.pill_height)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self._state = OverlayState.RECORDING
        self._smoothed_amplitude = 0.55
        self._wave_phase = 0.0
        self._proc_phase = 0.0
        self._pulse_alpha = 255
        self._pulse_dir = -1
        self._hover_gear = False
        self._processing_text = "Processing speech..."
        self._error_text = "No speech detected"

        self._num_bars = 7
        self._bar_heights = [6.0, 12.0, 22.0, 28.0, 20.0, 14.0, 8.0]

        # 60 FPS animation timer for live preview
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_anim_frame)
        self._anim_timer.start()

    def set_theme(self, theme: str) -> None:
        """Switch preview theme."""
        valid_themes = {"classic", "neon_glass", "matte_island", "fluent_mica"}
        if theme not in valid_themes:
            theme = "classic"
        self.theme = theme
        self.update()

    def set_preview_state(self, state: OverlayState, message: str = "") -> None:
        """Switch preview state (RECORDING, PROCESSING, ERROR, STANDBY)."""
        self._state = state
        if message:
            if state == OverlayState.ERROR:
                self._error_text = message
            elif state == OverlayState.PROCESSING:
                self._processing_text = message
        self.update()

    def _on_anim_frame(self) -> None:
        """Advance animation phases for realistic voice-reactive preview."""
        self._wave_phase += 0.12
        if self._wave_phase > 100000.0:
            self._wave_phase = 0.0

        self._proc_phase += 0.08
        if self._proc_phase > 100000.0:
            self._proc_phase = 0.0

        self._pulse_alpha += self._pulse_dir * 5
        if self._pulse_alpha <= 100:
            self._pulse_alpha = 100
            self._pulse_dir = 1
        elif self._pulse_alpha >= 255:
            self._pulse_alpha = 255
            self._pulse_dir = -1

        synth_amp = 0.42 + 0.38 * math.sin(self._wave_phase * 0.45)
        self._smoothed_amplitude = max(0.1, min(1.0, synth_amp))

        for i in range(self._num_bars):
            factor = math.sin(self._wave_phase * 1.4 + i * 0.6)
            h = 4.0 + (self._smoothed_amplitude * 24.0) * (0.35 + 0.65 * (factor + 1.0) / 2.0)
            self._bar_heights[i] = max(3.0, min(32.0, h))

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if not event:
            return
        in_gear = event.position().x() >= 324.0
        if in_gear != self._hover_gear:
            self._hover_gear = in_gear
            self.update()

    def leaveEvent(self, event: Any) -> None:
        if self._hover_gear:
            self._hover_gear = False
            self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        rect = QRectF(1.0, 1.0, float(self.width() - 2), float(self.height() - 2))
        self.paint_theme_overlay(painter, rect)
        painter.end()
