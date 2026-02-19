"""
app.py
EVK4 Event Camera Dashboard — Main Application Entry Point.
Wires together UI components, callbacks, and business logic.
"""
import os
import asyncio
from nicegui import ui, app
from core import AppState, load_config, RECONNECT_TIMEOUT
from ui import (
    build_main_layout,
    create_toggle_dark_callback,
    create_update_plots_callback,
    create_on_shape_drawn_callback,
    create_delta_t_change_callback,
    create_frames_change_callback,
    create_pick_file_callback,
    create_update_frame_display_callback,
    create_generate_frames_callback,
    create_export_frames_callback,
    create_time_range_slider_callback,
    create_time_range_input_callback,
    create_time_range_apply_callback,
    create_time_range_reset_callback,
    update_histogram_plot,
    update_iei_histogram,
    update_power_spectrum,
    update_timetrace,
)
from ui.callbacks import loading_overlay
from services import shutdown_executor
@ui.page('/')
async def main_page() -> None:
    """Build and wire the complete dashboard UI."""
    config = load_config()
    dark = ui.dark_mode(config["dark_mode"])
    state = AppState()
    class ComponentsHolder:
        components = None
    holder = ComponentsHolder()
    async def on_polarity_change_callback():
        if state.current_data is None or holder.components is None:
            return
        async with loading_overlay('Updating polarity mode...'):
            polarity_mode = holder.components.polarity_select.value
            update_histogram_plot(state, dark.value, polarity_mode, holder.components.histogram_plot)
            update_iei_histogram(state, dark.value, polarity_mode, holder.components.iei_plot)
            update_power_spectrum(state, dark.value, polarity_mode, holder.components.spectrum_plot)
            update_timetrace(state, dark.value, polarity_mode, holder.components.timetrace_plot)
            await asyncio.sleep(0.05)
    components = build_main_layout(dark, on_polarity_change=on_polarity_change_callback)
    holder.components = components
    toggle_dark = create_toggle_dark_callback(state, dark, components)
    update_plots = create_update_plots_callback(state, dark, components)
    on_shape_drawn = create_on_shape_drawn_callback(state, dark, components)
    on_delta_t_change = create_delta_t_change_callback(state, components)
    on_frames_change = create_frames_change_callback(state, components)
    handle_pick_file = create_pick_file_callback(state, dark, components)
    update_frame_display = create_update_frame_display_callback(state, components)
    generate_frames = create_generate_frames_callback(state, dark, components)
    export_frames = create_export_frames_callback(state, components)
    on_time_range_slider = create_time_range_slider_callback(state, components)
    on_time_range_input = create_time_range_input_callback(state, components)
    apply_time_range = create_time_range_apply_callback(state, dark, components)
    reset_time_range = create_time_range_reset_callback(state, dark, components)
    components.icon.on('click', toggle_dark)
    components.histogram_plot.on('plotly_relayout', on_shape_drawn)
    components.delta_t_input.on('update:model-value', on_delta_t_change)
    components.frames_input.on('update:model-value', on_frames_change)
    components.frame_slider.on('update:model-value', update_frame_display, throttle=0.1)
    components.time_range_slider.on('update:model-value', on_time_range_slider, throttle=0.05)
    components.time_range_from.on('update:model-value', on_time_range_input)
    components.time_range_to.on('update:model-value', on_time_range_input)
    components.open_file_btn.on('click', handle_pick_file)
    components.generate_frames_btn.on('click', generate_frames)
    components.export_frames_btn.on('click', export_frames)
    components.time_range_apply_btn.on('click', apply_time_range)
    components.time_range_reset_btn.on('click', reset_time_range)
def shutdown() -> None:
    """Clean shutdown: close the thread pool and exit."""
    print('Shutting down...')
    shutdown_executor()
    os._exit(0)
app.on_startup(lambda: app.native.main_window.maximize())
app.on_shutdown(shutdown)
ui.run(native=True, reconnect_timeout=RECONNECT_TIMEOUT)