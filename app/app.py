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

filepath = "./files/"
result_filenames = []

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
    Предупреждение пользователя
    :параметр stype: тип ошибки (строка)
    ------------------
    User Warning
    :param stype: error type (string)
    """
    gr.Warning("Выберите файл для распознавания!") if stype == 'file' else gr.Warning("Выберите модель распознавания!")


def inf():
    """
    Информация для пользователя, инструкция по работе с системой (так как я не смог вызвать функцию с параметром в виде строки с кнопки)
    ------------------
    Information for the user, instructions for working with the system (since I could not call the function with a param in the form of a string from the button)
    """
    gr.Info("Для начала работы - загрузите ваши файлы в формате jpg, jpeg, png")


def inform(stype: str):
    """
    Информация для пользователя, инструкция по работе с системой
    :параметр stype: тип информации (строка)
    ------------------
    User information, instructions for working with the system
    :param stype: information type (string)
    """
    if stype == 'working':
        gr.Info("Распознавание может занять некоторое время")
    elif stype == 'end':
        gr.Info("Посмотреть обработанные файлы можно, нажав на кнопку ниже")
    else:
        gr.Info("Как ты это сделал? Ачивку в студию")
   

def shorten_filename(filename, max_length):
    """
    Сокращение имени файла до max_length символов и добавление «...» в конец
    :параметр filename: название оригинального файла (строка)
    :параметр max_length: максимальная длина названия файла (целое)
    ------------------
    Shorten file name to max_length characters and add "..." to the end
    :param filename: name of original file (string)
    :param max_length: maximum length of file name (integer)
    """
    if len(filename) > max_length:
        return filename[:max_length] + '...'
    return filename


def confidence_plots(stats_df):
    """
    Создание инфографики на основе средних значений достоверности
    :параметр stats_df: pd.Dataframe с результатами детекции
    :return fig: график plotly
    ------------------
    Creating infographics based on average confidence values
    :stats_df param: pd.Dataframe with detection results
    :return fig: plotly graph
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
        paper_bgcolor='#0f172a',
        plot_bgcolor='#0f172a',
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
    Создание инфографики на основе детекции объектов на кадрах видео
    :параметр detections: list с результатами детекции
    :параметр detections: list с результатами детекции
    :return fig: график plotly
    ------------------
    Creating infographics based on average confidence values
    :stats_df param: pd.Dataframe with detection results
    :return fig: plotly graph
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
    
    # Enhance plot formatting
    fig.update_layout(
        xaxis_title='Время (сек)',
        yaxis_title='Количество объектов',
        hovermode='x unified',
        legend_title_text='Найденные классы',
        xaxis=dict(rangeslider=dict(visible=True)),
        height=500
    )
    
    return fig


def fileOpen():
    """
    Переход к папке последней детекции через проводник системы
    ------------------
    Go to the last detection folder via the system explorer
    """
    webbrowser.open(os.path.realpath(str(newfolder)))


def photoProcessing(files, cv_model):
    """
    Проверка предоставленных пользователем фотографий
    :параметр files: пути до файлов (массив строк)
    :параметр cv_model: номер выбранной модели (целое)
    :return detections_files, df, confidence_plots: изображения с bbox, таблица результатов, графики распределения
    ------------------
    Checking user-provided photos
    :param files: paths to files (array of strings)
    :param cv_model: name of the selected model (string)
    :return detections_files, df, confidence_plots: images with bbox, results table, distribution graphs
    """
    if files is not None and cv_model is not None:
        inform('working')
        time.sleep(1)
        
        folder_name = str(uuid.uuid4())
        output_folder = os.path.join('./files', folder_name)
        global newfolder 
        newfolder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        elapsed_times = pd.DataFrame(columns=['filename', 'model_type', 'processing_time'])
        detections_files = []
        global result_filenames
        result_filenames = []
        
        class_list = ['Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask']
        df = pd.DataFrame(columns=['file_name', 'class_id', 'class_name', 'confidence'])
        pd.set_option('display.precision', 3)
        
        for elem in files:
            
            filename, _ = os.path.splitext(elem.split('\\')[-1])
            result_filenames.append(filename)
            dfilename = os.path.join(output_folder, f'd_{filename}.jpg')
            detections_files.append(dfilename)
            
            detections_list, time_df  = img_ppe_detection(elem, dfilename, model_type=cv_model)
            elapsed_times = pd.concat([elapsed_times, time_df], ignore_index=True)
            
            img_detections = detections_list.confidence
            img_class_id = detections_list.class_id
            img_class_names = []
            
            if cv_model.find('YOLO'):
                img_class_names = detections_list.data['class_name']
            elif cv_model.find('RT'):
                for i in range(len(detections_list.class_id)):
                    img_class_names.append(class_list[detections_list.class_id[i]])

            for class_name, class_id, confidence in zip(img_class_names, img_class_id, img_detections):
                new_row = pd.DataFrame({
                    'file_name': [filename],
                    'class_id': [class_id],
                    'class_name': [class_name],
                    'confidence': [round(float(confidence), 3)]
                })
                df = pd.concat([df if not df.empty else None, new_row], ignore_index=True)
        
        detections_file_path = os.path.join(output_folder, 'Detections.csv')
        df.to_csv(detections_file_path, sep=',', index=False)
        
        # Shorten file names
        df['file_name'] = df['file_name'].apply(lambda x: shorten_filename(x, 20))
        
        # Unique object per image count
        unique_files = df['file_name'].unique()
        cross = pd.MultiIndex.from_product([unique_files, class_list], names=['file_name', 'class_name']).to_frame(index=False)
        counts = df.groupby(['file_name', 'class_name']).size().reset_index(name='count')
        result_df = cross.merge(counts, on=['file_name', 'class_name'], how='left').fillna({'count': 0})
        result_df['count'] = result_df['count'].astype(int)
        result_df = result_df[result_df['count'] > 0]
        result_df['class_name'] = pd.Categorical(result_df['class_name'], categories=class_list, ordered=True)
        result_df = result_df.sort_values(['file_name', 'class_name'])
        detections_file_path = os.path.join(output_folder, 'Statistics.csv')
        result_df.to_csv(detections_file_path, sep=',', index=False)
        result_df.rename(columns={'file_name': 'Изображение', 'class_name': 'Найденные классы', 'count': 'Количество'}, inplace=True)
        
        # Statistics calculation
        file_stats_df = df.groupby(['file_name', 'class_id', 'class_name'])['confidence'].agg(['mean', 'max', 'min', 'count']).reset_index()
        file_stats_df.columns = ['file_name', 'class_id', 'class_name', 'mean_confidence', 'max_confidence', 'min_confidence', 'detection_count']
        file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']] = file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']].round(3)
        file_stats_df = file_stats_df.sort_values(['file_name', 'mean_confidence'], ascending=[True, False])
        confidence_plot = confidence_plots(file_stats_df)
        inform('end')
        
        # Processing time validation
        elapsed_times['filename'] = elapsed_times['filename'].apply(lambda x: shorten_filename(x, 20))
        elapsed_times.rename(columns={'filename': 'Изображение', 'model_type': 'Модель', 'processing_time': 'Время (сек)'}, inplace=True)
                
        return detections_files, result_df, confidence_plot, elapsed_times
    
    elif cv_model is None:
        warning('radio')
        return None, None, None
    else:
        warning('file')
        return None, None, None


def videoProcessing(file, cv_model):
    """
    Проверка предоставленного пользователем видео
    :параметр file: путь до файла (строка)
    :параметр cv_model: номер выбранной модели (целое)
    :return dfilename, df: путь до видео с bbox, таблица результатов
    ------------------
    Checking user-provided video
    :param file: path to file (string)
    :param cv_model: number of selected model (integer)
    :return dfilename, df: path to video from bbox, results table
    """
    time.sleep(1)
    if file is not None and cv_model is not None:
        inform('working')
        
        folder_name = str(uuid.uuid4())
        output_folder = os.path.join('./files', folder_name)
        global newfolder 
        newfolder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
        filename, _ = os.path.splitext(file.split('\\')[-1])
        dfilename = os.path.join(output_folder, f'd_{filename}.mp4')
        dcsvfilename = os.path.join(output_folder, 'detections.csv')
        
        detection_data, time_df = video_ppe_detection(source=file, result_name=dfilename, output_folder=output_folder, model_type=cv_model)
        inform('end')
        time_df.rename(columns={'filename': 'Изображение', 'model_type': 'Модель', 'processing_time': 'Время (сек)'}, inplace=True)
        
        classes = ['Coverall', 'Face_Shield', 'Gloves', 'Googles', 'Mask']
        
        timeline_plot = timeline_plots(detection_data, classes)
        
        class_data = []
        
        all_classes = set()
        
        for frame in detection_data:
            all_classes.update(frame['class_counts'].keys())
        sorted_classes = sorted(all_classes)
        
        for frame in detection_data:
            time_entry = {
                'time': round(frame['timestamp'], 2)
            }
            for cls in sorted_classes:
                time_entry[f'{classes[cls]}'] = frame['class_counts'].get(cls, 0)
            class_data.append(time_entry)
        
        class_df = pd.DataFrame(class_data)
        class_df = class_df.reindex(columns=['time'] + [f'{classes[cls]}' for cls in sorted_classes])

        detections_file_path = os.path.join(output_folder, 'Detections.csv')
        class_df.to_csv(detections_file_path, sep=',', index=False)
        
        class_df.rename(columns={'time': 'Время (сек)',}, inplace=True)
        
        return dfilename, time_df, timeline_plot, class_df
    elif cv_model is None:
        warning('radio')
        return None, None
    else:
        warning('file')
        return None, None

def on_select(event_data: gr.SelectData):
    """
    Обновление названия файла после детекции
    ------------------
    Updating the file name after detection
    """
    return result_filenames[event_data.index]

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
        basic_info = gr.Button(value="О программе",visible=True, variant='huggingface')
    
    with gr.Row():
        with gr.Column():
            with gr.Tab('Изображение'):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        files_photo = gr.File(label="Загрузите ваши файлы здесь", file_types=['.png','.jpeg','.jpg'], file_count='multiple')                     
                    with gr.Column(scale=2):
                        gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br><br></p>
                                        <ul><font size="3px">
                                        <liа изображений для проверки;</li>
                                        <li>Обработка этих изображений моделью;</li>
                                        <li>Создание файлов результатов детекции в уникальной папке;</li>
                                        <li>Вывод интерактивной инфографики по каждому из изображений;</li>
                                        </ul></font>""", container=True)
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание", variant='secondary')
                        cv_model_img = gr.Dropdown(["YOLOv11-L", "RT-DETR-L", "RF-DETR-BASE"], value="YOLOv11-L", label="Модель", container=False)

                with gr.Row():
                    with gr.Column('Результат обработки'):
                        with gr.Row():
                            with gr.Column(variant='panel'):
                                predictImage = gr.Gallery(type="filepath", label="Предсказание модели", columns=[2], rows=[1], preview=True, allow_preview=True, object_fit="contain", height=500)
                                img_name = gr.Markdown(container=True, show_copy_button=True, min_height=50)
                            predictImageClass = gr.DataFrame(label="Результаты обработки", headers=[" "], elem_id='dataframe', max_height=570, show_row_numbers=True, show_fullscreen_button=False, show_search='search', show_copy_button=True)
                        with gr.Row():
                            statsPLOT = gr.Plot(label='Распределение найденных классов', elem_id='stats-plot', container=True)
                            time_stats = gr.DataFrame(label="Время обработки", headers=[" "], show_row_numbers=True, show_search='search')
            
            """-------------------------------------------------------------------------------------------------------------------------------------------------"""
            
            with gr.Tab('Видео'):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        file_video = gr.File(label="Загрузите ваши файлы здесь", file_types=['.mp4','.mkv'], file_count = 'single', scale=7)
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
                        cv_model_vid = gr.Dropdown(["YOLOv11-L", "RT-DETR-L", ], value="YOLOv11-L", label="Модель", container=False) #"RF-DETR-BASE"
                with gr.Row(equal_height=True):
                    with gr.Column():
                        predictVideo = gr.Video(label="Обработанное видео", interactive=False)
                        video_time_stats = gr.DataFrame(label="Время обработки", headers=[" "], datatype=["str", "number"], show_row_numbers=True)
                    with gr.Column():
                        predictVideoClass = gr.DataFrame(label="Результаты обработки", headers=[" "], elem_id='dataframe', max_height=470, show_row_numbers=True, show_fullscreen_button=False, show_search='search', show_copy_button=True)
                        timeline_plot = gr.Plot(label="Хронология классов",)
                                                        
    with gr.Row(): 
        with gr.Row(): 
            clr_btn = gr.ClearButton([files_photo, predictImage, predictImageClass, statsPLOT, file_video, predictVideo, predictVideoClass, time_stats, video_time_stats, timeline_plot], value="Очистить поля",)
            data_folder = gr.Button(value="Посмотреть файлы",)
    
    with gr.Row():
        gr.Markdown("""<p align="center"><a href="https://github.com/AlexeyLunyakov"><nobr>Выполнил Луняков Алексей</nobr></a></p>""")
        
    predictImage.select(on_select, None, img_name)
    
    btn_photo.click(photoProcessing, inputs=[files_photo, cv_model_img], outputs=[predictImage, predictImageClass, statsPLOT, time_stats])
    btn_video.click(videoProcessing, inputs=[file_video, cv_model_vid], outputs=[predictVideo, video_time_stats, timeline_plot, predictVideoClass])
    data_folder.click(fileOpen)
    basic_info.click(inf)

demo.launch(allowed_paths=["/assets/"])
