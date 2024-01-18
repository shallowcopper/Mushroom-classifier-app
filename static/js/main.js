// static/js/main.js
function classifyMushroom() {
    var capDiameter = document.getElementById('capDiameter').value;
    var capShape = document.getElementById('capShape').value;
    var capSurface = document.getElementById('capSurface').value;
    var capColor = document.getElementById('capColor').value;
    var doesBruiseOrBleed = document.getElementById('doesBruiseOrBleed').value;
    var gillAttachment = document.getElementById('gillAttachment').value;
    var gillColor = document.getElementById('gillColor').value;
    var stemHeight = document.getElementById('stemHeight').value;
    var stemWidth = document.getElementById('stemWidth').value;
    var stemColor = document.getElementById('stemColor').value;
    var hasRing = document.getElementById('hasRing').value;
    var ringType = document.getElementById('ringType').value;
    var habitat = document.getElementById('habitat').value;
    var season = document.getElementById('season').value;

    // Send the data to the server for classification
    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({

            capDiameter: parseInt(capDiameter),
            capShape: parseInt(capShape),
            capSurface: parseInt(capSurface),
            capColor: parseInt(capColor),
            doesBruiseOrBleed: parseInt(doesBruiseOrBleed),
            gillAttachment: parseInt(gillAttachment),
            gillColor: parseInt(gillColor),
            stemHeight: parseInt(stemHeight),
            stemWidth: parseInt(stemWidth),
            stemColor: parseInt(stemColor),
            hasRing: parseInt(hasRing),
            ringType: parseInt(ringType),
            habitat: parseInt(habitat),
            season: parseInt(season)
        }),
    })
    .then(response => response.text())
    .then(data => {
        console.log('Received data:', data);

        try {
            const jsonData = JSON.parse(data);
            document.getElementById('result').innerText = 'Result: ' + jsonData.result;
        } catch (error) {
            console.error('Error parsing JSON:', error);
        }
    })

}
