import os
import uuid
import time
import numpy as np
import gradio as gr
import pandas as pd
import webbrowser, os
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from processing import *

HOME = os.getcwd()

CLASS_LIST = ['Coverall', 'Face_Shield', 'Gloves', 'Googles', 'Mask']

custom_theme = gr.themes.Monochrome(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="slate",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('JetBrains Mono'), gr.themes.GoogleFont('Limelight'), 'sans-serif'],
).set(
    block_radius='*radius_xxl',
    button_large_radius='*radius_xl',
    button_large_text_size='*text_md',
    button_small_radius='*radius_xl',
)


def warning(stype: str):
    """
    User Warning
    
    :param stype: error type (string)
    
    ------------------
    Предупреждение пользователя
    
    :param stype: тип ошибки (строка)
    """
    gr.Warning("Выберите файл для распознавания!") if stype == 'file' else gr.Warning("Выберите модель распознавания!")


def inf():
    """
    Information for the user, instructions for working with the system
    ------------------
    Информация для пользователя, инструкция по работе с системой
    """
    gr.Info("Для начала работы - загрузите ваши файлы в формате jpg, jpeg, png")


def inform(stype: str):
    """
    User information, instructions for working with the system
    
    :parameter stype: information type (string)

    ------------------
    Информация для пользователя, инструкция по работе с системой
    
    :parameter stype: тип информации (строка)
    """
    if stype == 'working':
        gr.Info("Распознавание может занять некоторое время")
    elif stype == 'end':
        gr.Info("Посмотреть обработанные файлы можно, нажав на кнопку ниже")
    else:
        gr.Info("Как ты это сделал? Ачивку в студию")
   

def shorten_filename(filename, max_length):
    """
    Shorten file name to max_length characters and add "..." to the end
    
    :parameter filename: name of original file (string)
    :parameter max_length: maximum length of file name (integer)
    
    ------------------
    Сокращение имени файла до max_length символов и добавление «...» в конец
    
    :parameter filename: название оригинального файла (строка)
    :parameter max_length: максимальная длина названия файла (целое)
    
    
    """
    if len(filename) > max_length:
        return filename[:max_length] + '...'
    return filename


def confidence_plots(stats_df):
    """
    Creating infographics based on average confidence values
    
    :parameter stats_df: pd.Dataframe with detection results
    :return fig: plotly graph
    
    ------------------
    Создание инфографики на основе средних значений достоверности
    
    :parameter stats_df: pd.Dataframe с результатами детекции
    :return fig: график plotly
    """
    file_names = stats_df['file_name'].unique()
    fig = make_subplots(rows=1, cols=len(file_names), 
                        subplot_titles=file_names,
                        horizontal_spacing=0.05)
    
    for i, file_name in enumerate(file_names, start=1):
        file_data = stats_df[stats_df['file_name'] == file_name]
        bar = go.Bar(x=file_data['class_name'], 
                     y=file_data['mean_confidence'],
                     text=file_data['mean_confidence'],
                     textposition='auto',
                     name=file_name,
                     marker_color='#0A7153')
        
        fig.add_trace(bar, row=1, col=i)
        fig.update_traces(width=0.35)
        fig.update_yaxes(range=[0, 1], row=1, col=i)
    
    fig.update_layout(
        height = 400, 
        width = 400 * len(file_names),
        title={
            'text': "Средний уровень достоверности по классам",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        autosize=True,
        paper_bgcolor='rgba(0, 0, 0, 0.3)',
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
        font=dict(color='#ffffff'),
        xaxis=dict(tickangle=45),
    )
    
    fig.update_xaxes(
        gridcolor='#3f3f3f',
        zerolinecolor='#3f3f3f',
    )
    fig.update_yaxes(
        gridcolor='#3f3f3f',
        zerolinecolor='#3f3f3f',
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(color='#ffffff')
    
    return fig


def timeline_plots(detections: list, class_names: list = None):
    """
    Creating infographics based on average confidence values
    
    :parameter detections: list with detections results
    :parameter class_names: list with detection classes
    :return fig: plotly graph
    
    ------------------
    Создание инфографики на основе детекции объектов на кадрах видео
    
    :parameter detections: list с результатами детекции
    :parameter class_names: list с классами детекции
    :return fig: график plotly  
    """
    plot_data = []
    for frame in detections:
        timestamp = frame['timestamp']
        for cls_id, count in frame['class_counts'].items():
            plot_data.append({
                'Timestamp': timestamp,
                'Count': count,
                'Class': f'{class_names[cls_id]}'
            })

    if not plot_data:
        return px.line(title="No Detections Found")

    df = pd.DataFrame(plot_data)
    
    fig = px.line(
        df,
        x='Timestamp',
        y='Count',
        color='Class',
        title=' ',
        labels={'Count': 'Number of Detections'},
        hover_data=['Class', 'Count'],
        template='plotly_white'
    )
    
    fig.update_layout(
        autosize=True,
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
        font=dict(color='#ffffff'),
        xaxis=dict(tickangle=45),
    )
    
    fig.update_layout(
        xaxis_title='Время (сек)',
        yaxis_title='Количество объектов',
        hovermode='x unified',
        legend_title_text='Найденные классы',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    
    return fig


def fileOpen(folder_path):
    """
    Last detection folder via the system explorer
    ------------------
    Переход к папке последней детекции через проводник системы
    """
    webbrowser.open(os.path.realpath(str(folder_path)))


def generate_statistics(detections_df, class_list):
    """
    Statistics generation
    
    :parameter detections_df: model detections dataframe (dataframe)
    :parameter class_list: detection classes list (list)
    :return sorted stats_df: sorted statictics dataframe (dataframe)
    
    ------------------
    Генерация статистики

    :parameter detections_df: модель обнаружения dataframe (dataframe)
    :parameter class_list: список классов обнаружения (list)
    :return sorted stats_df: отсортированный statictics dataframe (dataframe)
    """
    cross_tab = pd.crosstab(
        index=detections_df['file_name'],
        columns=detections_df['class_name'],
        dropna=False
    ).reindex(columns=class_list, fill_value=0)
    
    stats_df = cross_tab.reset_index() \
        .melt(id_vars='file_name', value_vars=class_list, 
              var_name='class_name', value_name='count')
    
    stats_df = stats_df[stats_df['count'] > 0]
    stats_df['class_name'] = pd.Categorical(stats_df['class_name'], 
                                          categories=class_list, 
                                          ordered=True)
    return stats_df.sort_values(['file_name', 'class_name'])

RESULT_FILENAMES = []

def photoProcessing(files, model_type):
    """
    Checking user-provided photos
    
    :parameter files: paths to files (array of strings)
    :parameter model_type: name of the selected model (string)
    :return detections_files, df, confidence_plots: images with bbox, results table, distribution graphs
    
    ------------------
    Проверка предоставленных пользователем фотографий
    
    :parameter files: пути до файлов (массив строк)
    :parameter model_type: номер выбранной модели (целое)
    :return detections_files, df, confidence_plots: изображения с bbox, таблица результатов, графики распределения
    """
    if not files or model_type is None:
        warning('file' if model_type is not None else 'radio')
        return None, None, None, None
    
    inform('working')
    output_folder = os.path.join(HOME, 'files', str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    
    all_detections = []
    elapsed_times = []
    detections_files = []
    
    is_yolo = 'YOLO' in model_type
    is_rt = 'RT' in model_type
    is_rf = 'RF' in model_type
    
    for file_path in files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        RESULT_FILENAMES.append(filename)
        
        output_path = os.path.join(output_folder, f'd_{filename}.jpg')
        detections_files.append(output_path)
        
        detections, time_df = img_ppe_detection(file_path, output_path, model_type=model_type)
        elapsed_times.append({
            'filename': filename,
            'model_type': model_type,
            'processing_time': time_df['processing_time'].iloc[0]
        })
        
        confidences = detections.confidence
        class_ids = detections.class_id
        
        if is_yolo:
            class_names = detections.data['class_name']
        elif is_rt:
            class_names = [CLASS_LIST[cid] for cid in class_ids]
        elif is_rf:
            class_names = [CLASS_LIST[cid-1] for cid in class_ids]
            
        for i in range(len(class_ids)):
            all_detections.append({
                'file_name': filename,
                'class_id': class_ids[i],
                'class_name': class_names[i],
                'confidence': round(float(confidences[i]), 3)
            })
    
    detections_df = pd.DataFrame(all_detections)
    detections_df.to_csv(os.path.join(output_folder, 'Detections.csv'), index=False)
    
    stats_df = generate_statistics(detections_df, CLASS_LIST)
    stats_df.to_csv(os.path.join(output_folder, 'Statistics.csv'), index=False)
    
    file_stats_df = detections_df.groupby(['file_name', 'class_id', 'class_name'])['confidence'] \
        .agg(['mean', 'max', 'min', 'count']) \
        .rename(columns={'mean': 'mean_confidence', 'max': 'max_confidence', 
                         'min': 'min_confidence', 'count': 'detection_count'}) \
        .reset_index()
    file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']] = \
        file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']].round(3)
    
    confidence_plot = confidence_plots(file_stats_df)
    
    time_df = pd.DataFrame(elapsed_times)
    time_df['filename'] = time_df['filename'].apply(lambda x: shorten_filename(x, 20))
    time_df.rename(columns={
        'filename': 'Изображение',
        'model_type': 'Модель',
        'processing_time': 'Время (сек)'
    }, inplace=True)
    
    report_df = stats_df.rename(columns={
        'file_name': 'Изображение',
        'class_name': 'Найденные классы',
        'count': 'Количество'
    })[['Изображение', 'Найденные классы', 'Количество']]
    
    inform('end')
    return detections_files, report_df, confidence_plot, time_df, output_folder


def videoProcessing(file, model_type):
    """
    Checking user-provided video
    
    :param file: path to file (string)
    :param model_type: number of selected model (integer)
    :return dfilename, df: path to video from bbox, results table
    
    ------------------
    Проверка предоставленного пользователем видео
    
    :param file: путь до файла (строка)
    :param model_type: номер выбранной модели (целое)
    :return dfilename, df: путь до видео с bbox, таблица результатов  
    """
    if not file or model_type is None:
        warning('file' if model_type is not None else 'radio')
        return None, None, None, None
    
    inform('working')
    
    output_folder = os.path.join(HOME, 'files', str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(file))[0]
    output_video = os.path.join(output_folder, f'd_{filename}.mp4')
    detection_data, time_df = video_ppe_detection(
        source=file,
        result_name=output_video,
        output_folder=output_folder,
        model_type=model_type
    )

    timeline_plot = timeline_plots(detection_data, CLASS_LIST)
    
    class_indices = list(range(len(CLASS_LIST)))
    class_columns = ['time'] + CLASS_LIST
    class_data = []
    
    for frame in detection_data:
        row = [round(frame['timestamp'], 2)]
        row.extend(frame['class_counts'].get(cls, 0) for cls in class_indices)
        class_data.append(row)
    
    class_df = pd.DataFrame(class_data, columns=class_columns)
    class_df.to_csv(os.path.join(output_folder, 'Detections.csv'), index=False)
    
    time_df.rename(columns={
        'filename': 'Изображение',
        'model_type': 'Модель',
        'processing_time': 'Время (сек)'
    }, inplace=True)
    
    report_df = class_df.rename(columns={'time': 'Время (сек)'})
    
    inform('end')
    return output_video, time_df, timeline_plot, report_df, output_folder


def on_select(event_data: gr.SelectData):
    """
    Обновление названия файла после детекции
    ------------------
    Updating the file name after detection
    """
    return RESULT_FILENAMES[event_data.index]


custom_css = """
#theme-toggle {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    width: 30px;
    height: 30px;
}
#theme-toggle-2 {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    width: 30px;
    height: 30px;
}
#stats-plot {
    overflow-y: auto !important;
    overflow-x: auto !important;
    scrollbar-width: thin !important;
}
"""


def theme(new_theme):
    """
    Изменение темы светлая\темная
    :параметр new_theme: путь до файла (строка)
    :return toggle_theme: переключение темы
    ------------------
    Change theme light\dark
    :param new_theme: path to file (string)
    :return toggle_theme: switch theme
    """
    if new_theme == 'dark':
        toggle_theme = """
        function refresh() {
            const url = new URL(location);
            url.searchParams.set('__theme', 'dark')
            location.href = url.href;
        }
        """
    elif new_theme == 'light':
        toggle_theme = """
        function refresh() {
            const url = new URL(location);
            url.searchParams.set('__theme', 'light')
            console.log(url.searchParams.get('__theme'));
            location.href = url.href;
        }
        """
    return toggle_theme


theme_btns = """
    function hide_btns() {
        const url = new URL(location);
        /*console.log(url.searchParams.get('__theme'));*/
        if (url.searchParams.get('__theme') === "dark") {
            var x = document.getElementById("theme-toggle");
            /*console.log('dark-btn-hide');*/
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }
        }
        else if (url.searchParams.get('__theme') === "light") {
            var x = document.getElementById("theme-toggle-2");
            /*console.log('light-btn-hide');*/
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }
        }
    }
"""


with gr.Blocks(theme=custom_theme, css=custom_css, js=theme_btns, title='SAFE-MACS') as demo:

    toggle_button = gr.Button("☾", elem_id='theme-toggle')
    toggle_button.click(fn=None, inputs=None, outputs=None, js=theme('dark'))
    toggle_button = gr.Button("☀", elem_id='theme-toggle-2')
    toggle_button.click(fn=None, inputs=None, outputs=None, js=theme('light'))
    
    gr.Markdown("""<a name="readme-top"></a>
                    <h1 style="text-align:center;line-height: 0.3;" ><font size="30px"><strong style="font-family: Limelight">SAFE-MACS</strong></font></h1>
                    <p style="text-align:center;color:#0FC28F;line-height: 0.1;font-weight: bold;">Safety Automated Medical Control System</p>
                    <p align="center"><font size="4px">Интеллектуальная система цифрового автоконтроля<br>применения средств индивидуальной защиты медицинского персонала<br></font></p>
                    <p align="center"></p>""")
    with gr.Sidebar(width=400, position='right', open=False):
        gr.Markdown("""
                    <h1 align="center"><font size="4px">Боковое меню<br></font></h1>
                    """)
        basic_info = gr.Button(
            value="О программе",
            visible=True, 
            variant='huggingface'
        )
    
    with gr.Row():
        with gr.Column():
            with gr.Tab('Изображение'):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        files_photo = gr.File(
                            label="Загрузите ваши файлы здесь", 
                            file_types=['.png','.jpeg','.jpg'], 
                            file_count='multiple'
                        )                     
                    with gr.Column(scale=2):
                        gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br><br></p>
                                        <ul><font size="3px">
                                        <li>Загрузка изображений для проверки;</li>
                                        <li>Обработка этих изображений моделью;</li>
                                        <li>Создание файлов результатов детекции в уникальной папке;</li>
                                        <li>Вывод интерактивной инфографики по каждому из изображений;</li>
                                        </ul></font>""", container=True)
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание", variant='secondary')
                        model_type_img = gr.Dropdown(
                            ["YOLOv11-L", "RT-DETR-L", "RF-DETR-BASE"], 
                            value="YOLOv11-L", 
                            label="Модель", 
                            container=False
                        )

                with gr.Row():
                    with gr.Column('Результат обработки'):
                        with gr.Row():
                            with gr.Column(variant='panel'):
                                predictImage = gr.Gallery(
                                    type="filepath", 
                                    label="Предсказание модели", 
                                    columns=[2], 
                                    rows=[1], 
                                    preview=True, 
                                    allow_preview=True, 
                                    object_fit="contain", 
                                    height=500
                                )
                                img_name = gr.Markdown(
                                    container=True, 
                                    show_copy_button=True, 
                                    min_height=50
                                )
                            with gr.Column(variant='default'):
                                folder_name = gr.Markdown(
                                    container=True, 
                                    show_copy_button=True, 
                                    min_height=50, 
                                    render=False
                                )
                                predictImageClass = gr.DataFrame(
                                    label="Результаты обработки", 
                                    headers=[" "], 
                                    elem_id='dataframe', 
                                    max_height=570, 
                                    show_row_numbers=True, 
                                    show_fullscreen_button=False, 
                                    show_search='search', 
                                    show_copy_button=True
                                )
                        with gr.Row():
                            statsPLOT = gr.Plot(
                                label='Распределение найденных классов', 
                                elem_id='stats-plot', 
                                container=True
                            )
                            time_stats = gr.DataFrame(
                                label="Время обработки", 
                                headers=[" "], 
                                show_row_numbers=True, 
                                show_search='search'
                            )
            
            with gr.Tab('Видео'):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        file_video = gr.File(
                            label="Загрузите ваши файлы здесь", 
                            file_types=['.mp4','.mkv'], 
                            file_count = 'single', 
                            scale=7
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br><br></p>
                                    <ul><font size="3px">
                                    <li>Загрузка видео для проверки;</li>
                                    <li>Обработка этого видео моделью;</li>
                                    <li>Создание файлов результатов детекции в уникальной папке;</li>
                                    <li>Вывод интерактивной инфографики по каждому из классов;</li>
                                    </ul></font>""", container=True)
                with gr.Column():
                    with gr.Row(): 
                        btn_video = gr.Button(value="Начать распознавание",)
                        model_type_vid = gr.Dropdown(
                            ["YOLOv11-L", "RT-DETR-L", ], 
                            value="YOLOv11-L", 
                            label="Модель", 
                            container=False
                        )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        predictVideo = gr.Video(
                            label="Обработанное видео", 
                            interactive=False
                        )
                        video_time_stats = gr.DataFrame(
                            label="Время обработки", 
                            headers=[" "], 
                            datatype=["str", "number"], 
                            show_row_numbers=True
                        )
                    with gr.Column():
                        predictVideoClass = gr.DataFrame(
                            label="Результаты обработки", 
                            headers=[" "], 
                            elem_id='dataframe', 
                            max_height=470, 
                            show_row_numbers=True, 
                            show_fullscreen_button=False, 
                            show_search='search', 
                            show_copy_button=True
                        )
                        timeline_plot = gr.Plot(label="Хронология классов",)
                                                        
    with gr.Row(): 
        with gr.Row(): 
            clr_btn = gr.ClearButton(
                [files_photo, predictImage, predictImageClass, 
                 statsPLOT, file_video, predictVideo, 
                 predictVideoClass, time_stats, video_time_stats, 
                 timeline_plot, img_name], 
                value="Очистить поля",)
            data_folder = gr.Button(value="Посмотреть файлы",)
    
    with gr.Row():
        gr.Markdown("""<p align="center"><a href="https://github.com/AlexeyLunyakov"><nobr>Created solo by Alexey Lunyakov</nobr></a></p>""")
        
    predictImage.select(on_select, None, img_name)
    
    btn_photo.click(
        photoProcessing, 
        inputs=[files_photo, model_type_img],
        outputs=[predictImage, predictImageClass, statsPLOT, time_stats, folder_name]
    )
    btn_video.click(
        videoProcessing, 
        inputs=[file_video, model_type_vid], 
        outputs=[predictVideo, video_time_stats, timeline_plot, predictVideoClass, folder_name]
    )
    data_folder.click(
        fileOpen, 
        inputs=folder_name
    )
    basic_info.click(inf)

demo.launch(allowed_paths=["/assets/"])
