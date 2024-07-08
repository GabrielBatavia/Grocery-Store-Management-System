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
  const productContainer = document.getElementById("productContainer");
  const addProductBtn = document.getElementById("addProductBtn");

  // Function to calculate total for each product
  function calculateTotal(productGroup) {
    const priceInput = productGroup.querySelector(".price");
    const quantityInput = productGroup.querySelector(".quantity");
    const totalInput = productGroup.querySelector(".total");

    const price = parseFloat(priceInput.value);
    const quantity = parseInt(quantityInput.value);

    if (!isNaN(price) && !isNaN(quantity)) {
      totalInput.value = (price * quantity).toFixed(2);
    } else {
      totalInput.value = "";
    }
  }

  // Add event listener for each price and quantity input to calculate total
  function addCalculationEventListeners(productGroup) {
    const priceInput = productGroup.querySelector(".price");
    const quantityInput = productGroup.querySelector(".quantity");

    priceInput.addEventListener("input", () => calculateTotal(productGroup));
    quantityInput.addEventListener("input", () => calculateTotal(productGroup));
  }

  // Function to add a new product group
  function addProductGroup() {
    const productGroup = document.createElement("div");
    productGroup.className = "product-group";

    productGroup.innerHTML = `
      <div class="form-group">
        <label for="product">Product</label>
        <input type="text" class="product" name="product" required />
      </div>
      <div class="form-group">
        <label for="price">Price</label>
        <input type="number" class="price" name="price" required />
      </div>
      <div class="form-group">
        <label for="quantity">Quantity</label>
        <input type="number" class="quantity" name="quantity" required />
      </div>
      <div class="form-group">
        <label for="total">Total</label>
        <input type="number" class="total" name="total" readonly />
      </div>
    `;

    productContainer.appendChild(productGroup);
    addCalculationEventListeners(productGroup);
  }

  addProductBtn.addEventListener("click", addProductGroup);

  // Initialize the first product group
  const initialProductGroup = document.querySelector(".product-group");
  addCalculationEventListeners(initialProductGroup);

  // Handle form submission
  const orderForm = document.getElementById("orderForm");
  if (orderForm) {
    orderForm.addEventListener("submit", (event) => {
      event.preventDefault();
      // Logic to handle order submission goes here
      alert("Order submitted successfully!");
    });
  }
});
