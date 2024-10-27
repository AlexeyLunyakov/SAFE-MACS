<a name="readme-top"></a>  

<div align="center">
  <p align="center">
    <h1 align="center">Safety Automated Medical Control System (SAFE-MACS)</h1>
  </p>

  <p align="center">
    <p><strong>Приложение для цифрового автоконтроля применения СИЗ медицинского персонала с помощью компьютерного зрения.</strong></p>
    <br /><br />
  </p>
</div>

**Содержание:**
- [Проблематика](#title1)
- [Описание решения](#title2)
- [Тестирование решения](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика</a></h3> 
Необходимо создать, с применением технологий искусственного интеллекта, MVP в виде программного решения для автоматического контроля применения средств индивидуальной защиты (СИЗ) медицинским персоналом.

Решение может использоваться для:
* автоматизации процесса мониторинга соблюдения норм безопасности в медицинских учреждениях;
* оптимизации бизнес-процессов в сфере медицинской логистики и управления персоналом.

В задаче рассматривается работа с реальными изображениями медицинских работников в рабочей обстановке, поэтому решение включает несколько ключевых задач обучения моделей:
* точное определение наличия и корректности использования СИЗ на изображении;
* идентификация типов СИЗ (маски, перчатки, халаты и др.) с помощью моделей компьютерного зрения.


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - **Использованные модели:**
    - **```Object Detection```**:
      - ultralytics/YOLOv11;
    - **```Object Classification```**:
      - ultralytics/YOLOv11 (но здесь будет что-то другое, честно);

**Обоснование выбора моделей:**
* здесь
* будет
* обоснование
* но
* позже

Ссылки на репозитории моделей:
   - [YOLOv11](https://github.com/ultralytics/ultralytics)
  
<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>



## <h3 align="start"><a id="title3">Тестирование решения</a></h3> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:

  **```Gradio + ML-models;```**

  <br />

<details>
  <summary> <strong><i> Тестирование моделей с минимальным приложением на Gradio:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell recommended**) через терминал последовательно выполнить следующие команды:

    - Клонирование репозитория:
    ```
    git clone https://github.com/AlexeyLunyakov/SAFE-MACS.git
    ```
    - Создание и активация виртуального окружения:
    ```
    cd ./SAFE-MACS
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей (CUDA 12.4 required):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip3 install -r requirements.txt
    ```
    - После установки зависимостей (3-5 минут) можно запустить Gradio:
    ```
    python ./app/app.py
    ```
    или с возможнностью автоматического перезапуска при возникновении ошибок:
    ```
    cd ./app/
    gradio app.py
    ```

</details> 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


<a name="readme-top"></a>
