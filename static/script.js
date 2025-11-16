document.addEventListener('DOMContentLoaded', () => {
    const artForm = document.getElementById('art-form');
    const contentInput = document.getElementById('content-img');
    const styleInput = document.getElementById('style-img');
    const contentPreview = document.getElementById('content-preview');
    const stylePreview = document.getElementById('style-preview');
    const contentDropZone = document.getElementById('content-drop-zone');
    const styleDropZone = document.getElementById('style-drop-zone');

    const styleWeightSlider = document.getElementById('style-weight-slider');
    const styleWeightValue = document.getElementById('style-weight-value');

    const generateBtn = document.getElementById('generate-btn');
    const resultArea = document.getElementById('result-area');
    const loadingSpinner = document.getElementById('loading-spinner');
    const statusText = document.getElementById('status-text');
    const resultImg = document.getElementById('result-img');
    // Slider interactivity
    styleWeightSlider.addEventListener('input', () => {
        styleWeightValue.textContent = `Style Weight: ${styleWeightSlider.value}`;
    });

    // Drag and drop logic
    function setupDropZone(dropZone, input, preview) {
        dropZone.addEventListener('click', () => input.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault(); //since default is to block drops
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault(); //prevents browser from opening the file
            dropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files; //files is a FileList object like array
            if (files.length > 0) {
                input.files = files;
                showPreview(files[0], preview, dropZone);
            }
        });
        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                showPreview(input.files[0], preview, dropZone);
            } else {
                showPreview(null, preview, dropZone);
            }
        });
    }

    function showPreview(file, preview) {
        const previewText = preview.parentElement.querySelector('.preview-text');
        if (!file) {
            preview.src = '';
            preview.style.display = 'none';
            previewText.style.display = 'flex';
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
            previewText.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    setupDropZone(contentDropZone, contentInput, contentPreview);
    setupDropZone(styleDropZone, styleInput, stylePreview);

    // Form submission with style weight
    artForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const contentFile = contentInput.files[0];
        const styleFile = styleInput.files[0];
        const styleWeight = styleWeightSlider.value;

        if (!contentFile || !styleFile) {
            alert('Please upload both a content and a style image.');
            return;
        }

        // UI while processing
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        resultArea.classList.remove('hidden');
        document.getElementById('loading-container').style.display = 'flex';
        statusText.textContent = 'Uploading images and starting the process...';
        resultImg.style.display = 'none';

        const formData = new FormData();
        formData.append('content_image', contentFile);
        formData.append('style_image', styleFile);
        formData.append('style_weight', styleWeight);

        try {
            // start generating
            const response = await fetch('/generate', {
                method: 'POST', body: formData
            });

            if(!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server responded with an error: ${response.status}. Response: ${errorText}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // poll for the result
            pollForStatus(data.task_id);

        } catch (error) {
            statusText.textContent = `Error: ${error.message}`;
            loadingSpinner.style.display = 'none';
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate AI Art';
            document.getElementById('download-container').style.display = 'none';
        }
    });

    function pollForStatus(taskId) {
        statusText.textContent = 'The AI is painting... This can take a few minutes.';

        const interval = setInterval(async () => {
            try {
                const response = await fetch(`status/${taskId}`);
                const data = await response.json();

                if (data.status === 'complete') {
                    clearInterval(interval);
                    document.getElementById('loading-container').style.display = 'none';
                    resultImg.src = data.image_url;
                    resultImg.style.display = 'block';
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate AI Art';
                    
                    // Setup download button
                    const downloadContainer = document.getElementById('download-container');
                    const downloadBtn = document.getElementById('download-btn');
                    downloadBtn.href = data.image_url;
                    downloadContainer.style.display = 'block';
                } else if (data.status === 'error') {
                    clearInterval(interval);
                    statusText.textContent = 'An error occurred during generation.';
                    document.getElementById('loading-container').style.display = 'none';
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate AI Art';
                }
                // if status is processing we just wait and poll again
            } catch(error) {
                clearInterval(interval);
                statusText.textContent = `Error checking status: ${error.message}`;
                loadingSpinner.style.display = 'none';
                generateBtn.disabled = false;
                generateBtn.textContent = 'Generate AI Art';
            }
        }, 5000); //poll every 5 seconds
    }

});