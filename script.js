// =======================
// AUTH CHECK (Protect index.html)
// =======================
function checkAuth() {
    const user = sessionStorage.getItem("loggedInUser");
    if (!user) {
        window.location.href = "login.html";
    }
}

// =======================
// SIGNUP FUNCTION
// =======================
function signup() {
    const username = document.getElementById("su_username").value.trim();
    const password = document.getElementById("su_password").value.trim();
    const msg = document.getElementById("signupMsg");

    if (username === "" || password === "") {
        msg.style.color = "red";
        msg.innerText = "All fields are required";
        return;
    }

    if (localStorage.getItem(username)) {
        msg.style.color = "red";
        msg.innerText = "User already exists";
        return;
    }

    localStorage.setItem(username, password);

    msg.style.color = "lime";
    msg.innerText = "Signup successful! Redirecting to login...";

    setTimeout(() => {
        window.location.href = "login.html";
    }, 1500);
}

// =======================
// LOGIN FUNCTION
// =======================
function login() {
    const username = document.getElementById("username").value.trim();
    const password = document.getElementById("password").value.trim();
    const msg = document.getElementById("loginMsg");

    if (username === "" || password === "") {
        msg.style.color = "red";
        msg.innerText = "Enter username and password";
        return;
    }

    const storedPassword = localStorage.getItem(username);

    if (storedPassword === null) {
        msg.style.color = "red";
        msg.innerText = "User not found";
        return;
    }

    if (storedPassword !== password) {
        msg.style.color = "red";
        msg.innerText = "Invalid password";
        return;
    }

    sessionStorage.setItem("loggedInUser", username);

    msg.style.color = "lime";
    msg.innerText = "Login successful! Redirecting...";

    setTimeout(() => {
        window.location.href = "index.html";
    }, 1000);
}

// =======================
// LOGOUT FUNCTION
// =======================
function logout() {
    sessionStorage.removeItem("loggedInUser");
    window.location.href = "login.html";
}

// =======================
// STOCK PREDICTION
// =======================
function predictStock() {
    const stockInput = document.getElementById("stockInput");
    const stock = stockInput.value.trim().toUpperCase();

    const emptyMsg = document.getElementById("emptyMessage");
    const predictionBox = document.getElementById("predictionBox");

    // If input is empty
    if (stock === "") {
        emptyMsg.style.display = "block";
        predictionBox.style.display = "none";
        alert("Enter stock symbol");
        return;
    }

    // Input entered → hide empty message
    emptyMsg.style.display = "none";

    fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stock: stock })   // ✅ send stock symbol
    })
    .then(res => res.json())
    .then(data => {
        predictionBox.style.display = "block";

        const decisionElem = document.getElementById("decision");
        decisionElem.innerText = `Prediction for ${stock}: ${data.decision}`;
        decisionElem.style.color = data.decision === "BUY" ? "lime" : "red";

        document.getElementById("confidence").innerText =
            `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

        // ---- Display factors nicely ----
        let jsonText = "{\n";
        for (let key in data.factors) {
            jsonText += `  "${key}": "${data.factors[key]}",\n`;
        }
        jsonText = jsonText.slice(0, -2) + "\n}";

        document.getElementById("factorsList").innerText = jsonText;
    })
    .catch(() => {
        alert("Prediction failed");
        predictionBox.style.display = "none";
    });
}


// =======================
// CLEAR OUTPUT
// =======================
function clearResult() {
    document.getElementById("stockInput").value = "";
    document.getElementById("predictionBox").style.display = "none";
    document.getElementById("decision").innerText = "";
    document.getElementById("confidence").innerText = "";
    document.getElementById("factorsList").innerText = "";
}
