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
  
    const addProductBtn = document.getElementById("addProductBtn");
    if (addProductBtn) {
      addProductBtn.addEventListener("click", () => {
        window.location.href = "add_product.html";
      });
    }
  
    // Retrieve product data from sessionStorage and display it in the product list
    const productList = document.getElementById("productList");
    const savedProducts = JSON.parse(sessionStorage.getItem("products")) || [];
  
    savedProducts.forEach((product) => {
      const productItem = document.createElement("div");
      productItem.className = "product-item";
      productItem.textContent = `${product.name} - ${product.price} - ${product.quantity}`;
      productList.appendChild(productItem);
    });
  
    // Handle form submission
    const orderForm = document.getElementById("orderForm");
    if (orderForm) {
      orderForm.addEventListener("submit", (event) => {
        event.preventDefault();
        // Logic to handle order submission goes here
        alert("Order submitted successfully!");
        window.location.href = "index.html";
      });
    }
  });
  