# **AI Art Generator - Neural Style Transfer**

This is a web application that uses deep learning to merge the content of one image with the artistic style of another. It's built with a Python (Flask + PyTorch) backend and an HTML/CSS/JavaScript frontend.

The backend uses an asynchronous task queue to handle the long-running (3-5 minute) style transfer process without timing out the user's web request.

## **Tech Stack**

* Backend: Python, Flask, PyTorch (for the VGG-19 model), Pillow (PIL)

* Frontend: HTML, CSS, vanilla JavaScript (using fetch for AJAX)

* Core AI: Neural Style Transfer (NST) using a pre-trained VGG-19 model.

## **Features**

* Upload separate "Content" and "Style" images.

* Adjustable "Style Intensity" slider to control the final output.

* Asynchronous backend task processing to handle long-running AI jobs.

* Frontend polling to check the status of the art generation.

* Image preview before and after generation.

## **How to Run This Project Locally**

1. Clone the Repository

    ```
    git clone https://github.com/Njn03/AI-art-generator.git
    cd AI-art-generator
    ```


2. Create a Virtual Environment

    ```
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
    ```
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install Dependencies

    Install all the required Python libraries using the requirements.txt file.

    ```
    pip install -r requirements.txt
    ```  

4. Run the Flask Server
    ```
    python app.py
    ```
    The server will start, usually on http://127.0.0.1:5000.



5. Open the App

    Open your web browser and navigate to http://127.0.0.1:5000. You can now upload your images and generate AI art!
