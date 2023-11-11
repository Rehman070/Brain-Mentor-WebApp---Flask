async function CallAPI() {
    let result = await fetch("http://localhost:3000/Doctors");
    result = await result.json();
    console.log(result);

    const userBox = document.getElementById('userBox');
    
    result.forEach((user) => {
        const userBoxItem = document.createElement('div');
        userBoxItem.classList.add('Box');
        userBoxItem.innerHTML = `
            <img src=${user.image}></img>
            <h2>${user.name}</h2>
            <span>${user.profession}</span>
            <div class="social-links-doctors">
                <a href="#"><i class="fab fa-facebook-f"></i></a>
                <a href="#"><i class="fab fa-twitter"></i></a>
                <a href="#"><i class="fab fa-linkedin-in"></i></a>
                <a href="#"><i class="fab fa-google-plus"></i></a>
            </div>
            <button class="view-details-btn">View Profile</button>
        `;

        // Add a hover effect to the userBoxItem when it's clicked
        userBoxItem.addEventListener('click', () => {
            userBoxItem.classList.toggle('hovered');
        });

        userBox.appendChild(userBoxItem);
    });
}

CallAPI();