const title = document.querySelector(".title");
const button = document.querySelector(".submit-button");
const outputArea = document.querySelector(".output-area");

setTimeout(async (_) => {
  try {
    const response = await fetch("/name", {
      method: "get",
      headers: {
        "Content-Type": "application/json",
      },
    });
    title.innerHTML = (await response.json()).name;
  } catch (err) {
    alert(`Error: ${err}`);
  }
});

button.addEventListener("click", async (_) => {
  try {
    inputString = document.querySelector(".input-textarea").value;
    const response = await fetch("/", {
      method: "post",
      body: JSON.stringify([inputString]),
      headers: {
        "Content-Type": "application/json",
      },
    });
    console.log(response);
    outputArea.innerHTML = (await response.json()).response;
    outputArea.removeAttribute("hidden");
  } catch (err) {
    alert(`Error: ${err}`);
  }
});
