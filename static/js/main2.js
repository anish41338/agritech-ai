document
    .getElementById("upload-form")
    .addEventListener("submit", function (e) {
      e.preventDefault();

      const file = document.getElementById("file").files[0]; // Use the correct ID for the file input
      const formData = new FormData();
      formData.append("file", file); // Append file (image or video) to FormData

      // Send the file (image or video) to the backend
      fetch("/upload_file", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          // Display the predictions
          const resultsDiv = document.getElementById("results");
          resultsDiv.textContent = JSON.stringify(data, null, 2); // Show the prediction results
        })
        .catch((error) => console.error("Error:", error));
    });