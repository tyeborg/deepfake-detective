const processBtn = document.querySelector(".processBtn");
//processBtn.classList.add("btnLoading");
//processBtn.classList.remove("btnLoading");

window.addEventListener("load", () => {
    processBtn.classList.add("btnLoading");

    processBtn.addEventListener("transitionend", () => {
        processBtn.classList.remove("btnLoading");
    })
})


// ------------------------------------------------------------ //
// Initialize variables for form switching by button clicks.
const imageForm = document.querySelector("form.img-upload")
const videoForm = document.querySelector("form.video-upload")
const imageBtn = document.querySelector("label.image-toggle")
const videoBtn = document.querySelector("label.video-toggle")

// Switch from imageForm to videoForm upon button click.
videoBtn.onclick = (() => {
    imageForm.style.marginLeft = "-50%";

});
// Switch from videoForm to imageForm upon button click.
imageBtn.onclick = (() => {
    imageForm.style.marginLeft = "0%";
});

// ------------------------------------------------------------ //

const inputImg = document.querySelector("#imgUploadBtn");
const inputVid = document.querySelector("#videoUploadBtn");
const imgAreaDisplay = document.querySelector("#displayImg");
const videoAreaDisplay = document.querySelector("#displayVideo");

inputVid.addEventListener('change', function () {
    const videoFile = this.files[0];

    const videoErrorArea = document.querySelector("#videoErrorArea");
    const videoUploadedArea = document.querySelector("#videoUploadedArea");
    const videoProcessBtn = document.querySelector("#videoProcessBtn");

    // Initialize a list of all the valid file extensions
    let validExtensions = ['video/mp4', 'video/mov'];

    validateFile(videoFile, validExtensions, videoUploadedArea, videoErrorArea, videoProcessBtn);

    if(validExtensions.includes(videoFile.type)) {
        // Display the image in the 'upload image' area.
        displayVideo(videoFile);
    }
});

inputImg.addEventListener('change', function () {
    const image = this.files[0];

    const imgErrorArea = document.querySelector("#imgErrorArea");
    const imgUploadedArea = document.querySelector("#imgUploadedArea");
    const imgProcessBtn = document.querySelector("#imgProcessBtn");

    // Initialize a list of all the valid file extensions
    let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];

    validateFile(image, validExtensions, imgUploadedArea, imgErrorArea, imgProcessBtn);

    if(validExtensions.includes(image.type)) {
        // Display the image in the 'upload image' area.
        displayImg(image);
    }
});

// Construct a function that validates a file with a corresponding file extension.
function validateFile(file, validExtensions, uploadedArea, errorArea, processBtn) {
    // Ensure the right file type was uploaded
    // Initialize a variable defining the file type of the uploaded file
    let fileType = file.type;
    let altFileType = fileType.split("/")[1];

    if(validExtensions.includes(fileType)) {
        // Setting selected file name
        let fileName = file.name;
        // Initialize a variable for the file size.
        let fileSize = convertBytes(file.size);
        
        // Alter the filename depending if it is too long or not.
        if(fileName.length >= 7) {
            let splitName = fileName.split('.');
            fileName = splitName[0].substring(0, 7) + "... ." + splitName[1];
        }

        // Clear the progress area, in case the user uploads additional files
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
        const allImg = videoAreaDisplay.querySelectorAll('video');
        allImg.forEach(item => item.remove());
        
        // Read the newly uploaded video.
        const vidUrl = reader.result;
        const vid = document.createElement('video');

        vid.src = vidUrl
        // Add the image to the image area when user has uploaded it.
        videoAreaDisplay.appendChild(vid);
        vid.controls = true;
        vid.load();
        //vid.play();
        videoAreaDisplay.classList.add('active');
    }
    reader.readAsDataURL(videoFile)
}

function displayImg(image) {
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
    reader.readAsDataURL(image)
}

// Create a function that determines the size of a file.
function convertBytes(num) {
    // Initialize variables.
    let diff = num < 0;
    let units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

    if (diff) {
        num = -num;
    }
    if (num < 1) {
        return (diff ? '-' : '') + num + ' B';
    }

    let exponent = Math.min(Math.floor(Math.log(num) / Math.log(1000)), units.length - 1);
    num = Number((num / Math.pow(1000, exponent)).toFixed(2));
    let unit = units[exponent];

    return (diff ? '-' : '') + num  + ' ' + unit;
}