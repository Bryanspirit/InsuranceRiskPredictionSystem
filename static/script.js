document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm")
  const resultsSection = document.getElementById("resultsSection")
  const errorSection = document.getElementById("errorSection")

  form.addEventListener("submit", async (e) => {
    e.preventDefault()

    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]')
    const btnText = submitBtn.querySelector(".btn-text")
    const loading = submitBtn.querySelector(".loading")

    submitBtn.disabled = true
    btnText.style.display = "none"
    loading.style.display = "inline"

    // Hide previous results/errors
    resultsSection.style.display = "none"
    errorSection.style.display = "none"

    try {
      // Collect form data
      const formData = new FormData(form)
      const data = {}

      for (const [key, value] of formData.entries()) {
        // Convert numeric fields
        if (
          key.includes("Value") ||
          key.includes("Income") ||
          key.includes("Premium") ||
          key.includes("Amount") ||
          key.includes("Number") ||
          key.includes("Months")
        ) {
          data[key] = Number.parseFloat(value) || 0
        } else if (key === "Response") {
          data[key] = Number.parseInt(value)
        } else {
          data[key] = value
        }
      }

      console.log("Sending data:", data)

      // Send prediction request
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })

      const result = await response.json()

      if (response.ok) {
        displayResults(result)
      } else {
        displayError(result.error || "An error occurred during prediction")
      }
    } catch (error) {
      displayError("Network error: " + error.message)
    } finally {
      // Reset button state
      submitBtn.disabled = false
      btnText.style.display = "inline"
      loading.style.display = "none"
    }
  })
})

function displayResults(result) {
  const resultsSection = document.getElementById("resultsSection")
  const riskLevel = document.getElementById("riskLevel")
  const confidence = document.getElementById("confidence")
  const lowRiskBar = document.getElementById("lowRiskBar")
  const highRiskBar = document.getElementById("highRiskBar")
  const lowRiskPercent = document.getElementById("lowRiskPercent")
  const highRiskPercent = document.getElementById("highRiskPercent")

  // Update risk level
  riskLevel.textContent = result.risk_level
  riskLevel.className = result.prediction === 1 ? "risk-level high-risk" : "risk-level low-risk"

  // Update confidence
  confidence.textContent = `Confidence: ${result.confidence}%`

  // Update probability bars
  lowRiskBar.style.width = `${result.probability_low}%`
  highRiskBar.style.width = `${result.probability_high}%`
  lowRiskPercent.textContent = `${result.probability_low}%`
  highRiskPercent.textContent = `${result.probability_high}%`

  // Show results
  resultsSection.style.display = "block"
  resultsSection.scrollIntoView({ behavior: "smooth" })
}

function displayError(message) {
  const errorSection = document.getElementById("errorSection")
  const errorMessage = document.getElementById("errorMessage")

  errorMessage.textContent = message
  errorSection.style.display = "block"
  errorSection.scrollIntoView({ behavior: "smooth" })
}

function clearForm() {
  document.getElementById("predictionForm").reset()
  document.getElementById("resultsSection").style.display = "none"
  document.getElementById("errorSection").style.display = "none"
}

function fillSampleData() {
  // Fill with sample data for testing
  document.getElementById("customer_lifetime_value").value = "8500.75"
  document.getElementById("income").value = "65000"
  document.getElementById("monthly_premium_auto").value = "125.50"
  document.getElementById("months_since_last_claim").value = "18"
  document.getElementById("months_since_policy_inception").value = "36"
  document.getElementById("number_of_open_complaints").value = "1"
  document.getElementById("number_of_policies").value = "3"
  document.getElementById("total_claim_amount").value = "750.00"
  document.getElementById("response").value = "1"
  document.getElementById("coverage").value = "Extended"
  document.getElementById("education").value = "Master"
  document.getElementById("employment_status").value = "Employed"
  document.getElementById("marital_status").value = "Married"
  document.getElementById("location_code").value = "Suburban"
  document.getElementById("vehicle_class").value = "SUV"
  document.getElementById("vehicle_size").value = "Large"
}

// Check model status on page load
fetch("/model_info")
  .then((response) => response.json())
  .then((data) => {
    if (data.error) {
      displayError("Model not loaded. Please run the training script first.")
    } else {
      console.log("Model loaded successfully:", data)
    }
  })
  .catch((error) => {
    console.log("Could not check model status:", error)
  })
