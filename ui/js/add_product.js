document.addEventListener("DOMContentLoaded", () => {
    const productForm = document.getElementById("productForm");
  
    productForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const productName = document.getElementById("productName").value;
      const price = document.getElementById("price").value;
      const quantity = document.getElementById("quantity").value;
  
      const product = { name: productName, price, quantity };
      const products = JSON.parse(sessionStorage.getItem("products")) || [];
      products.push(product);
      sessionStorage.setItem("products", JSON.stringify(products));
  
      window.location.href = "order.html";
    });
  
    const cancelBtn = document.getElementById("cancelBtn");
    cancelBtn.addEventListener("click", () => {
      window.location.href = "order.html";
    });
  });
  