import os
import uuid
import time
import numpy as np
import gradio as gr
import pandas as pd
import webbrowser, os
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from processing import *

filepath = "./files/"

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


def create_confidence_plots(stats_df):
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
    :param cv_model: number of the selected model (integer)
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

        detections_files = []
        class_list = ['Coverall', 'Face_Shield', 'Gloves', 'Goggles', 'Mask']
        df = pd.DataFrame(columns=['file_name', 'class_id', 'class_name', 'confidence'])
        pd.set_option('display.precision', 3)
        
        for elem in files:
            
            filename, _ = os.path.splitext(elem.split('\\')[-1])
            dfilename = os.path.join(output_folder, f'd_{filename}.jpg')
            detections_files.append(dfilename)
            
            detections_list = img_ppe_detection(elem, dfilename, model_type=cv_model)
            img_detections = detections_list.confidence
            img_class_id = detections_list.class_id
            img_class_names = []
            
            if cv_model:
                img_class_names = detections_list.data['class_name']
            else: 
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
        
        detections_file_path = os.path.join(output_folder, 'detections.csv')
        df.to_csv(detections_file_path, sep=',', index=False)
        
        # Красивый вывод имен файлов
        df['file_name'] = df['file_name'].apply(lambda x: shorten_filename(x, 10))
        
        # Подсчет статистики
        file_stats_df = df.groupby(['file_name', 'class_id', 'class_name'])['confidence'].agg(['mean', 'max', 'min', 'count']).reset_index()
        file_stats_df.columns = ['file_name', 'class_id', 'class_name', 'mean_confidence', 'max_confidence', 'min_confidence', 'detection_count']
        file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']] = file_stats_df[['mean_confidence', 'max_confidence', 'min_confidence']].round(3)
        file_stats_df = file_stats_df.sort_values(['file_name', 'mean_confidence'], ascending=[True, False])
        confidence_plots = create_confidence_plots(file_stats_df)
        inform('end')
        return detections_files, df, confidence_plots
    
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
        dfilename = os.path.join(output_folder, f'd_{filename}.mp4v')
        dcsvfilename = os.path.join(output_folder, 'detections.csv')

        video_ppe_detection(source=file, result_name=dfilename, output_folder=output_folder, model_type=cv_model)
        inform('end')
        
        df = pd.read_csv(dcsvfilename)
        
        return dfilename, df
    elif cv_model is None:
        warning('radio')
        return None, None
    else:
        warning('file')
        return None, None


custom_css = """
#theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    width: 30px;
    height: 30px;
}
#theme-toggle-2 {
    position: fixed;
    top: 10px;
    right: 10px;
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


with gr.Blocks(theme=custom_theme, css=custom_css, js=theme_btns) as demo:

    toggle_button = gr.Button("☾", elem_id='theme-toggle')
    toggle_button.click(fn=None, inputs=None, outputs=None, js=theme('dark'))
    toggle_button = gr.Button("☀", elem_id='theme-toggle-2')
    toggle_button.click(fn=None, inputs=None, outputs=None, js=theme('light'))
    
    gr.Markdown("""<a name="readme-top"></a>
                    <h1 style="text-align:center;line-height: 0.3;" ><font size="30px"><strong style="font-family: Limelight">SAFE-MACS</strong></font></h1>
                    <p style="text-align:center;color:#0FC28F;line-height: 0.1;font-weight: bold;">Safety Automated Medical Control System</p>
                    <p align="center"><font size="4px">Интеллектуальная система цифрового автоконтроля<br>применения средств индивидуальной защиты медицинского персонала<br></font></p>
                    <p align="center"></p>""")

    with gr.Row():
        with gr.Column():
            with gr.Tab('Изображение'):
                with gr.Row(equal_height=True):
                    files_photo = gr.File(label="Загрузите ваши файлы здесь", file_types=['.png','.jpeg','.jpg'], file_count='multiple', scale=7)
                    cv_model_img = gr.Radio(["RT-DETR-L", "YOLOv11-M"], label="Модель", scale=1, type='index')
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание",)
                        tmp = 'end'
                        InfoImage = gr.Button(value="Подробнее")
                with gr.Row():
                        with gr.Row('Результат обработки', equal_height=True):
                            with gr.Column(variant='panel',):
                                predictImage = gr.Gallery(type="filepath", label="Предсказание модели", columns=[2], rows=[1], preview=True, allow_preview=True, object_fit="contain", height=500)
                                statsPLOT = gr.Plot(label='Распределение найденных классов', elem_id='stats-plot', container=True)
                            with gr.Column():
                                gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br></p>
                                            <ul><font size="3px">
                                            <li>Загрузка изображений для проверки;</li>
                                            <li>Обработка этих изображений моделью;</li>
                                            <li>Создание файлов результатов детекции в уникальной папке;</li>
                                            <li>Вывод интерактивной инфографики по каждому из изображений;</li>
                                            <li>Фиксирование СИЗ и уверенности модели в таблице для всех изображений;</li>
                                            </ul></font>""")
                                predictImageClass = gr.DataFrame(headers=["Результаты обработки"], elem_id='dataframe')
            with gr.Tab('Видео'):
                with gr.Row(equal_height=True):
                    file_video = gr.File(label="Загрузите ваши файлы здесь", file_types=['.mp4','.mkv'], file_count = 'single', scale=7)
                    cv_model_vid = gr.Radio(["RT-DETR-L", "YOLOv11-M"], label="Модель", scale=1, type='index')
                with gr.Column():
                    with gr.Row(): 
                        btn_video = gr.Button(value="Начать распознавание",)
                        InfoVid = gr.Button(value="Подробнее",)
                with gr.Row(equal_height=True):
                    with gr.Column(variant='panel',):
                        predictVideo = gr.Video(label="Обработанное видео", interactive=False)
                        hz = gr.Plot(label='Сделать график время-класс', elem_id='stats-plot-2', container=True)
                    with gr.Column():
                        gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br></p>
                                    <ul><font size="3px">
                                    <li>Загрузка видео для проверки;</li>
                                    <li>Обработка вашего видео моделью;</li>
                                    <li>Создание файлов результатов детекции в уникальной папке;</li>
                                    <li>Вывод интерактивной инфографики для найденных классов СИЗ;</li>
                                    <li>Фиксирование СИЗ и уверенности модели в таблице с таймингами;</li>
                                    </ul></font>""")
                        predictVideoClass = gr.DataFrame(headers=["Результаты обработки"], elem_id='dataframe')
                                                        
    with gr.Row(): 
        with gr.Row(): 
            clr_btn = gr.ClearButton([files_photo, predictImage, predictImageClass, statsPLOT, file_video, predictVideo, predictVideoClass, cv_model_img, cv_model_vid, InfoImage], value="Очистить контекст",)
            btn2 = gr.Button(value="Посмотреть файлы",)
    
    with gr.Row():
        gr.Markdown("""<p align="center">Выполнил Луняков Алексей, студент ИКБО-04-21</p>""")
        
    btn_photo.click(photoProcessing, inputs=[files_photo, cv_model_img], outputs=[predictImage, predictImageClass, statsPLOT,])
    btn_video.click(videoProcessing, inputs=[file_video, cv_model_vid], outputs=[predictVideo, predictVideoClass,])
    btn2.click(fileOpen)
    InfoImage.click(inf) 
    InfoVid.click(inf) 

demo.launch(allowed_paths=["/assets/"])
