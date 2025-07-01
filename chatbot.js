document.addEventListener("DOMContentLoaded", function () {
    let isOpen = false;
    const button = document.querySelector('.toggle-button');
    const overlay = document.querySelector('.popup-overlay');
    const notification = document.querySelector('.notification');
    const popupContent = document.querySelector('.popup-content');
    const iframe = document.querySelector('.popup-iframe');

    function showNotification() {
        if (!isOpen) {
            notification.classList.add('active');
        }
    }
    function hideNotification() {
        notification.classList.remove('active');
    }
    setTimeout(() => { showNotification(); }, 3000);

    window.togglePopup = function () {
        isOpen = !isOpen;
        button.classList.toggle('active');
        overlay.classList.toggle('active');
        if (isOpen) {
            hideNotification();
        } else {
            showNotification();
        }
    };

    window.closePopup = function (event) {
        if (event.target === overlay) {
            togglePopup();
        }
    };

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && isOpen) {
            togglePopup();
        }
    });

    popupContent.addEventListener('click', (event) => {
        event.stopPropagation();
    });

    iframe.addEventListener('error', function () {
        console.error('Error loading iframe content');
    });
});