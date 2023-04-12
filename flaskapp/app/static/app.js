// Initialize variables for form switching by button clicks.
const imageForm = document.querySelector("form.imgUpload")
const videoForm = document.querySelector("form.videoUpload")
const imageBtn = document.querySelector("label.imageToggle")
const videoBtn = document.querySelector("label.videoToggle")

// Switch from imageForm to videoForm upon button click.
videoBtn.onclick = (() => {
    imageForm.style.marginLeft = "-50%";

});
// Switch from videoForm to imageForm upon button click.
imageBtn.onclick = (() => {
    imageForm.style.marginLeft = "0%";
});

// Initialize variables to display the image or video submitted by the user.
const inputImg = document.querySelector("#imgUploadBtn");
const inputVid = document.querySelector("#videoUploadBtn");
const imgAreaDisplay = document.querySelector("#displayImg");
const videoAreaDisplay = document.querySelector("#displayVideo");

// Determine if the video submitted by the user is appropriate for processing.
inputVid.addEventListener('change', function () {
    // Obtain the file submitted by the user.
    const videoFile = this.files[0];

    // Initialize variables for the validation section and video process button.
    const videoErrorArea = document.querySelector("#videoErrorArea");
    const videoUploadedArea = document.querySelector("#videoUploadedArea");
    const videoProcessBtn = document.querySelector("#videoProcessBtn");

    // Initialize a list of all the valid file extensions.
    let validExtensions = ['video/mp4', 'video/mov'];

    validateFile(videoFile, validExtensions, videoUploadedArea, videoErrorArea, videoProcessBtn);

    // Determine if the input video submitted by the user is a 
    // video of an mp4 or mov type.
    if(validExtensions.includes(videoFile.type)) {
        // Display the video in the 'Upload Image' area.
        displayVideo(videoFile);
    }
});

// Determine if the image submitted by the user is appropriate for processing.
inputImg.addEventListener('change', function () {
    // Obtain the file submitted by the user.
    const image = this.files[0];

    // Initialize variables for the validation section and image process button.
    const imgErrorArea = document.querySelector("#imgErrorArea");
    const imgUploadedArea = document.querySelector("#imgUploadedArea");
    const imgProcessBtn = document.querySelector("#imgProcessBtn");

    // Initialize a list of all the valid image extensions.
    let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

    validateFile(image, validExtensions, imgUploadedArea, imgErrorArea, imgProcessBtn);

    // Determine if the file sent by the user is of a validated image file type.
    if(validExtensions.includes(image.type)) {
        // Display the image in the 'Upload Image' area.
        displayImg(image);
    }
});

// Construct a function that validates a file with a corresponding file extension.
function validateFile(file, validExtensions, uploadedArea, errorArea, processBtn) {
    // Ensure the right file type was uploaded.
    // Initialize a variable defining the file type of the uploaded file.
    let fileType = file.type;
    let altFileType = fileType.split("/")[1];

    if(validExtensions.includes(fileType)) {
        // Setting selected file name.
        let fileName = file.name;
        // Initialize a variable for the file size.
        let fileSize = convertBytes(file.size);
        
        // Alter the filename depending if it is too long or not.
        if(fileName.length >= 6) {
            let splitName = fileName.split('.');
            fileName = splitName[0].substring(0, 6) + "... ." + splitName[1];
        }

        // Clear the progress area, in case the user uploads additional files.
        errorArea.innerHTML = "";

        let uploadedHTML = `<li class="row">
                                <div class="content">
                                    <i class="uil uil-check-circle"></i>
                                    <div class="details">
                                        <span class="name">${fileName} • Uploaded</span>
                                    </div>
                                </div>
                                <span class="size">${fileSize}</span>
                            </li>`;
        
        // Update the area with the 'updated information'.                    
        uploadedArea.innerHTML = uploadedHTML;

        // Enable button here!
        processBtn.disabled = false;

    }else {
        // Display Error message in error section.
        let errorHTML = `<li class="row">
                            <div class="content">
                                <i class="uil uil-file-slash"></i>
                                <div class="details">
                                    <span class="error">Error: ${altFileType} file is not applicable</span>
                                </div>
                            </div>
                            <i class="uil uil-exclamation-circle"></i>
                        </li>`
        
        // Clear the progress area, in case the user uploads additional files
        uploadedArea.innerHTML = "";
        // Update the area with the 'error information'.
        errorArea.innerHTML = errorHTML;

        // Disable button here!
        processBtn.disabled = true;
    }
}

function displayVideo(videoFile) {
    // Make the newly uploaded video display on the upload area.
    const reader = new FileReader();
    reader.onload = () => {
        // Remove the previous item in the upload queue.
        const allVid = videoAreaDisplay.querySelectorAll('video');
        allVid.forEach(item => item.remove());
        
        // Read the newly uploaded video.
        const vidUrl = reader.result;
        const vid = document.createElement('video');

        vid.src = vidUrl
        // Add the video to the video area when user has uploaded it.
        videoAreaDisplay.appendChild(vid);
        vid.controls = true;
        vid.load();
        videoAreaDisplay.classList.add('active');
    }
    reader.readAsDataURL(videoFile)
}

function displayImg(imageFile) {
    // Make the newly uploaded image display on the upload image area.
    const reader = new FileReader();
    reader.onload = () => {
        // Remove the previous image in the upload queue.
        const allImg = imgAreaDisplay.querySelectorAll('img');
        allImg.forEach(item => item.remove());
        
        // Read the newly uploaded image.
        const imgUrl = reader.result;
        const img = document.createElement('img');
        img.src = imgUrl
        // Add the image to the image area when user has uploaded it.
        imgAreaDisplay.appendChild(img);
        imgAreaDisplay.classList.add('active');
    }
    reader.readAsDataURL(imageFile)
}

// Create a function that determines the size of a file.
function convertBytes(num) {
    // Initialize variables.
    let diff = num < 0;
    // List all of the potential file sizes.
    let units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    if (diff) {
        num = -num;
    }
    if (num < 1) {
        return (diff ? '-' : '') + num + ' B';
    }

    // Exact the file size.
    let exponent = Math.min(Math.floor(Math.log(num) / Math.log(1000)), units.length - 1);
    num = Number((num / Math.pow(1000, exponent)).toFixed(2));
    let unit = units[exponent];

    return (diff ? '-' : '') + num  + ' ' + unit;
}

// Construct a function that submits a form form once process button is clicked.
function formSubmit(processBtn, formType) {
    processBtn.addEventListener("click", function () {
        formType.submit();
    });
}

// Enable onclick functionality to Image Process button.
function handleImgFormSubmit() {
    const imgProcessBtn = document.querySelector("#imgProcessBtn");
    const imgForm = document.querySelector("#imgForm");

    // Submit Image form once Image process button is clicked.
    formSubmit(imgProcessBtn, imgForm);
}

// Enable onclick functionality to Video Process button.
function handleVideoFormSubmit() {
    const videoProcessBtn = document.querySelector("#videoProcessBtn");
    const videoForm = document.querySelector("#videoForm");

    // Submit Video form once Video process button is clicked.
    formSubmit(videoProcessBtn, videoForm);
}

// Construct a method that resets everything to its default
// functionality by clicking the 'RESET' button.
const resetBtn = document.querySelector("#resetBtn");
resetBtn.onclick = function() {
    const selectors = [
        "#displayImg > img:last-of-type",
        "#displayVideo > video:last-of-type",
        ".faceCaptureImg > img:last-of-type",
        ".faceCaptureImg > video:last-of-type",
        ".validationSection > li:last-of-type",
        ".classificationSection > li:last-of-type"
    ];

    // Iterate throughout the list to set everything to default.
    selectors.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
            element.remove();
        });
    });
};

// Create a function that add spinner effect to button once clicked.
function enableButtonSpinner(processBtn) {
    processBtn.addEventListener("click", function () {
        // Add spinner functionality once button is clicked.
        processBtn.classList.add("btnLoading");
    });
}

// Initialize the process buttons for the spinner effect.
const imgProcessBtn = document.querySelector("#imgProcessBtn");
const videoProcessBtn = document.querySelector("#videoProcessBtn");

// Enable spinning effect for Image and Video Process buttons once clicked.
enableButtonSpinner(imgProcessBtn);
enableButtonSpinner(videoProcessBtn);