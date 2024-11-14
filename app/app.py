import gradio as gr
import webbrowser, os
import time
import plotly.express as px
import pandas as pd
from processing import *

filepath="./files/"

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

def photoProcessing(file, ):
    time.sleep(1)
    print(file)
    if file is not None:
        info_req()
        detections_list = ppe_detection(file, './files/detections.jpg')
        class_names = detections_list.data['class_name']
        confidences = detections_list.confidence
        pd.set_option('display.precision', 3)
        df = pd.DataFrame({
            'class_name': class_names,
            'confidence': confidences
        })
        df['confidence'] = df['confidence'].astype('float64').round(3)
        info_res()
        return './files/detections.jpg', df
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

def fileOpen():
    webbrowser.open(os.path.realpath(filepath))
   
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
                file_photo = gr.File(label="Фотография", file_types=['.png','.jpeg','.jpg'])
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание",)
                        triggerImage = gr.Button(value="Подробнее",)
                with gr.Row():
                        with gr.Row('Результат обработки'):
                            with gr.Column():
                                predictImage = gr.Image(type="pil", label="Предсказание модели")
                            with gr.Column():
                                gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br></p>
                                            <ul><font size="3px">
                                            <li>Детекция СИЗ медицинского персонала;</li>
                                            <li>Найденные классы СИЗ медицинского персонала и уверенность модели (слева-направо);</li>
                                            </ul></font>""")
                                predictImageClass = gr.DataFrame(label="Полученные классы", headers=["Class Name", "Confidence"])
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
            clr_btn = gr.ClearButton([file_photo, predictImage, predictImageClass, ], value="Очистить контекст",)
            btn2 = gr.Button(value="Посмотреть файлы",)
    
    with gr.Row():
        gr.Markdown("""<p align="center">Выполнил Луняков Алексей, студент ИКБО-04-21</p>""")

    btn_photo.click(photoProcessing, inputs=[file_photo, ], outputs=[predictImage, predictImageClass,])
    btn_video.click(videoProcessing, inputs=[file_video, ], outputs=[predictVideo, predictVideoClass,])
    btn2.click(fileOpen)
    triggerImage.click(info_fn)
    triggerVideo.click(info_fn)

demo.launch(allowed_paths=["/assets/"])