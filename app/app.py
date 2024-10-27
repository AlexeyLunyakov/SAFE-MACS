import gradio as gr
import webbrowser, os
import time
import plotly.express as px
# from processing import *

filepath="./files/"

light_theme = gr.themes.Monochrome(
    primary_hue="red",
    secondary_hue="amber",
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
    primary_hue="lime",
    secondary_hue="blue",
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
        # sign_detection(file, './files/detection.jpg')
        # number, probability = sign_recognition('./files/detection.jpg', 'recognition.jpg')
        # string = f'Номер: {number}\n\nУверенность OCR-модели: {probability:.2f}%'
        info_res()
        return './files/bot_deer_2.jpg', './files/bot_deer_2.jpg', 'Здесь будут распознанные классы'
        # return './files/detection.jpg', './files/cropped_image_recognition.jpg', string
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
                    <p style="text-align:center;color:#de2c00;line-height: 0.1;font-weight: bold;">Safety Automated Medical Control System</p>
                    <p align="center"><font size="4px">Интеллектуальная система цифрового автоконтроля<br>применения средств индивидуальной защиты медицинского персонала<br></font></p>
                    <p align="center"></p>""")

    with gr.Row():
        with gr.Column():
            with gr.Tab('Детектирование и распознавание по фотографии'):
                file_photo = gr.File(label="Фотография", file_types=['.png','.jpeg','.jpg'])
                with gr.Column():
                    with gr.Row():
                        btn_photo = gr.Button(value="Начать распознавание",)
                        triggerImage = gr.Button(value="Подробнее",)
                with gr.Row():
                        with gr.Row('Результат обработки'):
                            with gr.Column():
                                predictImage = gr.Image(type="pil", label="Предсказание модели")
                                cropImage = gr.Image(type="pil", label="Обрезанный фрагмент")
                            with gr.Column():
                                gr.Markdown("""<p align="start"><font size="4px">Что происходит в данном блоке?<br></p>
                                            <ul><font size="3px">
                                            <li>Детекция СИЗ медицинского персонала (верняя картинка);</li>
                                            <li>Кроп изображения по bbox от YOLO (нижняя картинка);</li>
                                            <li>Распознание СИЗ медицинского персонала (текстовое поле ниже);</li>
                                            </ul></font>""")
                                predictImageClass = gr.Textbox(label="Полученные классы", placeholder="Здесь будет таблица данных по файлам", interactive=False, lines=7)
                                
    with gr.Row(): 
        with gr.Row(): 
            btn2 = gr.Button(value="Посмотреть файлы",)
            clr_btn = gr.ClearButton([file_photo, predictImage, cropImage, predictImageClass, ], value="Очистить контекст",)
    
    with gr.Row():
        gr.Markdown("""<p align="center">Выполнил Луняков Алексей, студент ИКБО-04-21</p>""")

    btn_photo.click(photoProcessing, inputs=[file_photo, ], outputs=[predictImage, cropImage, predictImageClass,])
    btn2.click(fileOpen)
    triggerImage.click(info_fn)

demo.launch(allowed_paths=["/assets/"])