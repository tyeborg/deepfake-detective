// Initialize variables.
const selectImage = document.querySelector('.select-image');
const inputFile = document.querySelector('#file');

// Enable the SELECT FILE button to access computer files upon a clic.
selectImage.addEventListener('click', function () {
    inputFile.click()
})
inputFile.addEventListener('change', function () {
    const image = this.files[0];
    console.log(image);
})

// Initialize variables.
const form = document.querySelector("form"),
fileInput = form.querySelector(".file-input"),
progressArea = document.querySelector(".progress-area"),
uploadedArea = document.querySelector(".uploaded-area");
errorArea = document.querySelector(".error-area");

// Enable file input to be activated upon a user's click.
form.addEventListener("click", () => {
    fileInput.click();
});

fileInput.onchange = ({target}) => {
    // Receive the first file the user uploads.
    let file = target.files[0]
    
    validateFile(file);  
}

function getFilename(file) {
    // Setting selected file name
    let fileName = file.name;

    // Determine if the filename exceeds 7 characters.
    if(fileName.length >= 7) {
        let splitName = fileName.split('.');
        // Modify the filename as its too long to showcase in the sections.
        fileName = splitName[0].substring(0, 7) + "... ." + splitName[1];
    }

    return fileName;
}

function validateFile(file) {
    // Ensure the right file type was uploaded.
    // Initialize a variable defining the file type of the uploaded file.
    let fileType = file.type;
    // Initialize a list of all the valid file extensions
    let validExtensions = ['video/mp4', 'video/mov', 'image/jpeg', 'image/jpg', 'image/png'];
    // Initialize a variable that will showcase the filename.
    let fileName = getFilename(file);

    // Determine if the uploaded file includes the mp4 or mov extension.
    if(validExtensions.includes(fileType)){
        // Calling uploadFile with passing file name as an argument
        uploadFile(file.name);
    }else {
        let errorHTML = `<li class="row">
                            <div class="content">
                                <i class="fas fa-file-circle-xmark"></i>
                                <div class="details">
                                    <span class="error">${fileName} is not an MP4 or MOV</span>
                                </div>
                            </div>
                            <i class="fas fa-circle-exclamation"></i>
                        </li>`

        // Clear the progress/uploaded area, in case the user uploads additional files
        progressArea.innerHTML = "";
        uploadedArea.innerHTML = "";
        errorArea.innerHTML = errorHTML;
    }
}

function uploadFile(name) {
    // Create new xml obj (AJAX)
    let xhr = new XMLHttpRequest(); 
    var url = 'upload.php'; 

    // Send post request to the specified URL/File
    xhr.open("POST", url, true);

    //Send the proper header information along with the request
    //xhr.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');
    
    xhr.upload.addEventListener("progress", ({loaded, total}) => {
        // Get percentage of loaded file size
        let fileLoaded = Math.floor((loaded / total) * 100);
        // Get file size in KB from bytes
        let fileTotal = Math.floor(total / 1000);
        let fileSize;

        // If file size is less than 1024 then add only KB else convert size from KB to MB
        (fileTotal < 1024) ? fileSize = fileTotal + " KB" : fileSize = (loaded / (1024 * 1024)).toFixed(2) + " MB";

        let progressHTML = `<li class="row">
                                <i class="fas fa-file-alt"></i>
                                <div class="content">
                                    <div class="details">
                                        <span class="name">${name} • Uploading</span>
                                        <span class="percent">${fileLoaded}%</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress" style="width: ${fileLoaded}%"></div>
                                    </div>
                                </div>
                            </li>`;
        
        uploadedArea.classList.add("onprogress");
        // Clear the progress area, in case the user uploads additional files
        uploadedArea.innerHTML = "";
        errorArea.innerHTML = "";
        // Set up the progress area after user selects file
        progressArea.innerHTML = progressHTML;
        
        // Set up the 'Completed Uploaded' area if file uploaded successfully
        if(loaded == total) {
            progressArea.innerHTML = "";
            let uploadedHTML = `<li class="row">
                                    <div class="content">
                                        <i class="fas fa-file-alt"></i>
                                        <div class="details">
                                            <span class="name">${name} • Uploaded</span>
                                            <span class="size">${fileSize}</span>
                                        </div>
                                    </div>
                                    <i class="fas fa-check"></i>
                                </li>`;

            uploadedArea.classList.remove("onprogress");
            uploadedArea.innerHTML = uploadedHTML;
            //uploadedArea.insertAdjacentHTML("afterbegin", uploadedHTML);
        }
    });
    // FormData is an object to easily send form data
    let data = new FormData(form);
    //data.append("file", name);
    // Sending form data to php
    xhr.send(data);
}