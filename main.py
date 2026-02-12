from nicegui import ui, app
from pathlib import Path
from utils import raw_to_npz, compute_event_histogram
import asyncio
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

@ui.page('/')
def main_page():
    dark = ui.dark_mode(True)
    current_file = None
    current_data = None

    def toggle_dark():
        dark.toggle()
        icon.set_name('light_mode' if dark.value else 'dark_mode')

        if histogram_card.visible:
            fig = histogram_plot.figure
            fig.update_layout(template='plotly_dark' if dark.value else 'plotly')
            histogram_plot.update()

    def update_histogram():
        if current_data is None:
            return
        
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        
        polarity = polarity_select.value
        if polarity == 'CD ON (polarity=1)':
            mode = 'on'
        elif polarity == 'CD OFF (polarity=0)':
            mode = 'off'
        else:
            mode = 'all'
        
        histogram = compute_event_histogram(events, width, height, mode)
        fig = go.Figure(go.Heatmap(
            z=histogram, 
            colorscale='Viridis',
            colorbar=dict(title='Count')
        ))
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            yaxis=dict(scaleanchor='x', scaleratio=1),
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_dark' if dark.value else 'plotly',
        )
        histogram_card.visible = True
        histogram_plot.figure = fig
        histogram_plot.update()

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
        nonlocal current_file, current_data
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
        
        current_data = np.load(current_file)
        events = current_data['events']
        width, height = int(current_data['width']), int(current_data['height'])
        
        bias_names = ['bias_diff', 'bias_diff_off', 'bias_diff_on', 'bias_fo', 'bias_hpf', 'bias_refr']
        bias_columns = []
        bias_row = {}
        for name in bias_names:
            if name in current_data:
                bias_columns.append({'name': name, 'label': name, 'field': name})
                bias_row[name] = int(current_data[name])
        
        if bias_columns:
            bias_table.columns = bias_columns
            bias_table.rows = [bias_row]
            bias_table.visible = True
        else:
            bias_table.visible = False

        duration = (events['t'][-1] - events['t'][0]) / 1e6
        event_count = len(events)
        event_rate = event_count / duration if duration > 0 else 0
        
        stats_table.columns = [
            {'name': 'events', 'label': 'Events', 'field': 'events'},
            {'name': 'duration', 'label': 'Duration', 'field': 'duration'},
            {'name': 'rate', 'label': 'Event rate', 'field': 'rate'},
            {'name': 'resolution', 'label': 'Resolution', 'field': 'resolution'},
        ]
        stats_table.rows = [{
            'events': f'{event_count:,}',
            'duration': f'{duration:.2f} s',
            'rate': f'{event_rate:,.0f} ev/s',
            'resolution': f'{width} x {height}',
        }]
        stats_table.visible = True

        update_histogram()

    with ui.header().classes('justify-between items-center'):
        ui.label('EVK4 Dashboard').classes('text-xl font-bold')
        icon = ui.icon('light_mode', size='md').classes('cursor-pointer')
        icon.on('click', toggle_dark)

    with ui.row().classes('w-full items-center gap-4'):
        ui.button('Open File', on_click=pick_file).classes('w-64')
        overwrite_toggle = ui.switch('OVERWRITE')
        file_label = ui.label().classes('text-gray-400')
        file_label.visible = False

    ui.separator()

    with ui.row().classes('w-full items-center gap-4'):
        stats_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})
        stats_table.visible = False

        bias_table = ui.table(columns=[], rows=[], column_defaults={'align': 'center', 'headerClasses': 'uppercase text-primary'})
        bias_table.visible = False
    
    with ui.card() as histogram_card:
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('EVENT HISTOGRAM').classes('text-lg font-bold')
            polarity_select = ui.select(
                options=['BOTH', 'CD ON (polarity=1)', 'CD OFF (polarity=0)'],
                value='BOTH',
                label='MODE',
                on_change=lambda: update_histogram()
            ).classes('w-48')
        histogram_plot = ui.plotly({})
    histogram_card.visible = False

app.on_startup(lambda: app.native.main_window.maximize())
ui.run(native=True)