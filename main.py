from nicegui import ui, app
from pathlib import Path
from utils import raw_to_npz
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

@ui.page('/')
def main_page():
    dark = ui.dark_mode(True)
    current_file = None

    def toggle_dark():
        dark.toggle()
        icon.set_name('light_mode' if dark.value else 'dark_mode')

    async def pick_file():
        result = await app.native.main_window.create_file_dialog(
            allow_multiple=False
        )
        if result and len(result) > 0:
            path = Path(result[0])
            if path.suffix.lower() in ('.raw', '.npz'):
                await process_file(path)
            else:
                ui.notify('Please select a .raw or .npz file', type='negative')

    async def process_file(path):
        nonlocal current_file
        suffix = path.suffix.lower()
        if suffix == '.raw':
            npz_path = path.with_suffix('.npz')
            
            if npz_path.exists() and not overwrite_toggle.value:
                ui.notify(f'{npz_path.name} already exists, loading existing file.', type='warning')
            else:
                overlay = ui.dialog().props('persistent')
                with overlay, ui.card().classes('items-center p-8'):
                    ui.spinner(size='xl')
                    ui.label(f'Converting {path.name}...').classes('mt-4')
                overlay.open()
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(executor, raw_to_npz, path, overwrite_toggle.value)
                
                overlay.close()
                ui.notify(f'Saved: {npz_path}')
            
            current_file = npz_path
        elif suffix == '.npz':
            current_file = path
            ui.notify(f'Loaded: {path.name}')
        
        file_label.text = f'Loaded: {current_file.name}'
        file_label.visible = True
        
        data = np.load(current_file)
        events = data['events']
        width, height = int(data['width']), int(data['height'])
        
        duration = (events['t'][-1] - events['t'][0]) / 1e6
        event_count = len(events)
        event_rate = event_count / duration if duration > 0 else 0
        
        bias_names = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']
        rows = []
        for name in bias_names:
            if name in data:
                rows.append({'name': name, 'value': int(data[name])})
        
        if rows:
            bias_table.rows = rows
            bias_table.visible = True
        else:
            bias_table.visible = False

        stats_table.rows = [
            {'stat': 'Events', 'value': f'{event_count:,}'},
            {'stat': 'Duration', 'value': f'{duration:.2f} s'},
            {'stat': 'Event rate', 'value': f'{event_rate:,.0f} ev/s'},
            {'stat': 'Resolution', 'value': f'{width} x {height}'},
        ]
        stats_table.visible = True

    with ui.header().classes('justify-between items-center'):
        ui.label('EVK4 Dashboard').classes('text-xl font-bold')
        icon = ui.icon('light_mode', size='md').classes('cursor-pointer')
        icon.on('click', toggle_dark)

    with ui.row().classes('w-full justify-between items-start'):
        with ui.row().classes('items-center gap-4'):
            ui.button('Open File', on_click=pick_file).classes('w-64')
            overwrite_toggle = ui.switch('Overwrite')
            file_label = ui.label().classes('text-gray-400')
            file_label.visible = False

        columns = [
            {'name': 'name', 'label': 'Bias', 'field': 'name'},
            {'name': 'value', 'label': 'Value', 'field': 'value'},
        ]
        bias_table = ui.table(columns=columns, rows=[], column_defaults={
            'align': 'left',
            'headerClasses': 'uppercase text-primary'})
        bias_table.visible = False

    with ui.row().classes('items-center gap-4'):
        stats_columns = [
            {'name': 'stat', 'label': 'Statistic', 'field': 'stat'},
            {'name': 'value', 'label': 'Value', 'field': 'value'},
        ]
        stats_table = ui.table(columns=stats_columns, rows=[])
        stats_table.visible = False

app.on_startup(lambda: app.native.main_window.maximize())
ui.run(native=True)