async function predict() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

        if (!file) {
        alert("Please select an image!");
        return;
    }

    // preview
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById("previewImg").src = e.target.result;
    };
    reader.readAsDataURL(file);

    // send to backend
    const formData = new FormData();
    formData.append("image", file);

    const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("argmax").innerText = data.argmax.label;
    document.getElementById("confidence").innerText =
        (data.argmax.confidence * 100).toFixed(2) + "%";

    const cp = data.conformal_prediction;
    const ul = document.getElementById("cpSet");
    ul.innerHTML = "";

    Object.entries(cp).forEach(([fruit, prob]) => {
    const li = document.createElement("li");
    li.innerText = `${fruit}: ${(prob * 100).toFixed(2)}%`;
    ul.appendChild(li);
    });
}
