import os
import uuid
import time
import gradio as gr
import pandas as pd
import webbrowser, os
import plotly.express as px

from processing import *

filepath = "./files/"

light_theme = gr.themes.Monochrome(
    primary_hue="teal",
    secondary_hue="emerald",
    neutral_hue="gray",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('JetBrains Mono'), gr.themes.GoogleFont('Limelight'), 'system-ui', 'sans-serif'],
).set(
    block_radius='*radius_xxl',
    button_large_radius='*radius_xl',
    button_large_text_size='*text_md',
    button_small_radius='*radius_xl',
)

dark_theme = gr.themes.Monochrome(
    primary_hue="green",
    secondary_hue="lime",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('Inter'), gr.themes.GoogleFont('Limelight'), 'system-ui', 'sans-serif'],
    neutral_hue="gray",
).set(
    block_radius='*radius_xxl',
    button_large_radius='*radius_xl',
    button_large_text_size='*text_md',
    button_small_radius='*radius_xl',
)

def warning_file():
    gr.Warning("Выберите файл для распознавания!")

def info_fn():
    gr.Info("Для начала работы - загрузите ваш файл в формате jpg, jpeg, png")

def info_req():
    # startup_conf()
    gr.Info("Распознавание может занять некоторое время")
    
def info_res():
    gr.Info("Посмотреть обработанные файлы можно, нажав на кнопку ниже")

def shorten_filename(filename, max_length):
    """Сокращение имени файла до max_length символов и добавление «...» в конец"""
    if len(filename) > max_length:
        return filename[:max_length] + '...'
    return filename

def fileOpen():
    webbrowser.open(os.path.realpath(str(newfolder)))

def photoProcessing(files, ):
    time.sleep(1)
    if files is not None:
        info_req()
        
        folder_name = str(uuid.uuid4())
        output_folder = os.path.join('./files', folder_name)
        global newfolder 
        newfolder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        detections_files = []
        df = pd.DataFrame(columns=['file_name', 'class_name', 'confidence'])
        pd.set_option('display.precision', 3)
        
        for elem in files:
            
            filename, _ = os.path.splitext(elem.split('\\')[-1])
            dfilename = os.path.join(output_folder, f'd_{filename}.jpg')
            detections_files.append(dfilename)
            
            detections_list = ppe_detection(elem, dfilename)
            img_class_names = detections_list.data['class_name']
            img_detections = detections_list.confidence
            
            for class_name, confidence in zip(img_class_names, img_detections):
                new_row = pd.DataFrame({
                    'file_name': [filename],
                    'class_name': [class_name],
                    'confidence': [round(float(confidence), 3)]
                })
                df = pd.concat([df, new_row], ignore_index=True)
        
        detections_file_path = os.path.join(output_folder, 'detections.csv')
        df.to_csv(detections_file_path, sep='\t', index=False)

        df['file_name'] = df['file_name'].apply(lambda x: shorten_filename(x, 10))
        
        info_res()
        return detections_files, df
    else:
        warning_file()
        return None, None

def videoProcessing(file, ):
    time.sleep(1)
    if file is not None:
        info_req()
        process_video(source=file, destination='result.mp4')
        info_res()
        with open('detections.csv', mode='r') as detect_file:
            string = detect_file.readlines()
        full_text = ''
        for el in string:
            el = el.replace('\n', '')
            data = el.split(',')
            full_text += 'Начало интервала: ' + data[0] + '; Конец интервала: ' + data[1] + '; Количество встреченных СИЗ: ' + data[2] + '\n'
        return 'result.mp4', full_text
    else:
        warning_file()
        return None, None, None


custom_css = """
#theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
}
"""

toggle_theme_js = """
function toggleTheme() {
    const themeToggle = document.getElementById('theme-toggle');
    const isDarkMode = document.body.classList.contains('dark');
    
    if (isDarkMode) {
        document.body.classList.remove('dark');
        document.body.classList.add('light');
        themeToggle.textContent = '☾';
    } else {
        document.body.classList.remove('light');
        document.body.classList.add('dark');
        themeToggle.textContent = '☀';
    }
}
"""

output = [gr.Dataframe(row_count = (4, "dynamic"), col_count=(4, "fixed"), label="Predictions")]

with gr.Blocks(theme=light_theme, css=custom_css) as demo:
    gr.HTML(
        """
        <button id="theme-toggle" onclick="toggleTheme()">🌙</button>
        <script>
        """ + toggle_theme_js + """
        </script>
        """
    )
    gr.Markdown("""<a name="readme-top"></a>
                    <h1 style="text-align:center;line-height: 0.3;" ><font size="30px"><strong style="font-family: Limelight">SAFE-MACS</strong></font></h1>
                    <p style="text-align:center;color:#0FC28F;line-height: 0.1;font-weight: bold;">Safety Automated Medical Control System</p>
                    <p align="center"><font size="4px">Интеллектуальная система цифрового автоконтроля<br>применения средств индивидуальной защиты медицинского персонала<br></font></p>
                    <p align="center"></p>""")

    with gr.Row():
        with gr.Column():
            with gr.Tab('Распознавание PPE по фотографии'):
                files_photo = gr.File(label="Фотография", file_types=['.png','.jpeg','.jpg'], file_count='multiple')
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание",)
                        triggerImage = gr.Button(value="Подробнее",)
                with gr.Row():
                        with gr.Row('Результат обработки'):
                            with gr.Column():
                                predictImage = gr.Gallery(type="filepath", label="Предсказание модели", columns=[2], rows=[1], preview=True, allow_preview=True, object_fit="contain", height=500)
                            with gr.Column():
                                gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br></p>
                                            <ul><font size="3px">
                                            <li>Детекция СИЗ медицинского персонала;</li>
                                            <li>Найденные классы СИЗ медицинского персонала и уверенность модели (слева-направо);</li>
                                            </ul></font>""")
                                predictImageClass = gr.DataFrame(label="Полученные классы", headers=["Image Name","Class Name", "Confidence"], max_height=380, elem_id='dataframe')
            with gr.Tab('Трекинг PPE по видео'):
                file_video = gr.File(label="Видео", file_types=['.mp4','.mkv'])
                with gr.Column():
                    with gr.Row(): 
                        btn_video = gr.Button(value="Начать распознавание",)
                        triggerVideo = gr.Button(value="Подробнее",)
                with gr.Row():
                    with gr.Tab('Результат обработки'):
                        with gr.Row():
                            predictVideo = gr.Video(label="Обработанное видео", interactive=False)
                            predictVideoClass = gr.Textbox(label="Результат обработки", placeholder="Здесь будут общие данные по файлу", interactive=False, lines=7)
                                                        
    with gr.Row(): 
        with gr.Row(): 
            clr_btn = gr.ClearButton([files_photo, predictImage, predictImageClass, ], value="Очистить контекст",)
            btn2 = gr.Button(value="Посмотреть файлы",)
    
    with gr.Row():
        gr.Markdown("""<p align="center">Выполнил Луняков Алексей, студент ИКБО-04-21</p>""")

    btn_photo.click(photoProcessing, inputs=[files_photo, ], outputs=[predictImage, predictImageClass,])
    btn_video.click(videoProcessing, inputs=[file_video, ], outputs=[predictVideo, predictVideoClass,])
    btn2.click(fileOpen)
    triggerImage.click(info_fn)
    triggerVideo.click(info_fn)

demo.launch(allowed_paths=["/assets/"])


'''
if files is not None:
        info_req()
        filenames, class_names, confidences = [], [], []
        df = pd.DataFrame({
            'file_name': filenames,
            'class_name': class_names,
            'confidence': confidences
        })
        pd.set_option('display.precision', 3)
        for elem in files:
            filename, file_extension = os.path.splitext(elem.split('\\')[-1])
            filenames.append(filename)
            detections_list = ppe_detection(elem, f'./files/d_{filename}.jpg')
            img_class_names = detections_list.data['class_name']
            img_detections = detections_list.confidence
            class_names.append(img_class_names)
            confidences.append(img_detections)
            
        df['confidence'] = df['confidence'].astype('float64').round(3)
        info_res()
        return files, df
'''