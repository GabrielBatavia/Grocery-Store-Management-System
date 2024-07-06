document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll(".btn");

  buttons.forEach((button) => {
    button.addEventListener("mouseover", () => {
      button.style.boxShadow = "0 6px 12px rgba(0, 123, 255, 0.4)";
      button.style.transition = "all 0.3s ease";
    });

    button.addEventListener("mouseout", () => {
      button.style.boxShadow = "0 4px 8px rgba(0, 123, 255, 0.3)";
      button.style.transition = "all 0.3s ease";
    });
  });

  // Event listener for New Order button
  const newOrderBtn = document.getElementById("newOrderBtn");
  if (newOrderBtn) {
    newOrderBtn.addEventListener("click", () => {
      window.location.href = "order.html";
    });
  }

  // Event listener for Manage Products button
  const manageProductsBtn = document.getElementById("manageProductsBtn");
  if (manageProductsBtn) {
    manageProductsBtn.addEventListener("click", () => {
      window.location.href = "manage_product.html";
    });
  }

  // Modal handling for managing products
  const modal = document.getElementById("productModal");
  const addNewProductBtn = document.getElementById("addNewProductBtn");
  const closeBtn = document.querySelector(".close");
  const closeModalBtn = document.getElementById("closeBtn");

  if (addNewProductBtn) {
    addNewProductBtn.addEventListener("click", () => {
      modal.style.display = "block";
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  }

  if (closeModalBtn) {
    closeModalBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  }

  window.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.style.display = "none";
    }
  });

  const productForm = document.getElementById("productForm");
  if (productForm) {
    productForm.addEventListener("submit", (event) => {
      event.preventDefault();
      // Logic to add new product goes here
      modal.style.display = "none";
    });
  }

  // Order handling
  const priceInput = document.getElementById("price");
  const quantityInput = document.getElementById("quantity");
  const totalInput = document.getElementById("total");
  const calculateTotalBtn = document.getElementById("calculateTotalBtn");

  if (calculateTotalBtn) {
    calculateTotalBtn.addEventListener("click", () => {
      const price = parseFloat(priceInput.value);
      const quantity = parseInt(quantityInput.value);

      if (!isNaN(price) && !isNaN(quantity)) {
        totalInput.value = (price * quantity).toFixed(2);
      } else {
        totalInput.value = "";
        alert("Please enter valid numbers for price and quantity.");
      }
    });
  }

  const orderForm = document.getElementById("orderForm");
  if (orderForm) {
    orderForm.addEventListener("submit", (event) => {
      event.preventDefault();
      // Logic to handle order submission goes here
      alert("Order submitted successfully!");
    });
  }
});
