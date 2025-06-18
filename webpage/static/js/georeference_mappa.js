const MAP_DATA_JSON = "static/maps.json";

let map = L.map('map', {
    attributionControl: false,
    zoomControl: false,
    minZoom: -4,
}).setView([37.52561664048445, 15.074417095820316], 15);

/*
var Stadia_OSMBright = L.tileLayer('https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.{ext}', {
	minZoom: 0,
	maxZoom: 24,
	attribution: '&copy; <a href="https://www.stadiamaps.com/" target="_blank">Stadia Maps</a> &copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
	ext: 'png'
}).addTo(map);
*/

var googleStreets = L.tileLayer('http://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',{
    maxZoom: 24,
    maxNativeZoom: 22,
    subdomains:['mt0','mt1','mt2','mt3']
}).addTo(map);

const markerIcon = L.divIcon({
  html: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#000000"><path d="M0 0h24v24H0z" fill="none"/><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>`,
  className: "marker",
  iconSize: [36, 36],
  iconAnchor: [18, 34],
});

const mapList = document.getElementById("map-select");
let mapData = [];

function refreshList() {
    fetch(MAP_DATA_JSON)
        .then(response => response.json())
        .then(data => {
            console.log(data)
            mapData = data;
            mapList.innerHTML = "";
            data.forEach(map => {
                const mapItem = document.createElement("option");
                mapItem.value = map.id;
                mapItem.textContent = map.id;
                mapList.appendChild(mapItem);
            });

            const selectedMap = mapList.value;
            const selectMapData = mapData.find(map => map.id === selectedMap);
            if (selectMapData) {
                console.log(selectMapData);
                showImage(selectMapData);
            }
        });
}

const refreshButton = document.getElementById("refresh-list");
refreshButton.addEventListener('click', refreshList);

refreshList();




mapList.addEventListener('change', function() {
    const selectedMap = mapList.value;
    const selectMapData = mapData.find(map => map.id === selectedMap);
    if (selectMapData) {
        console.log(selectMapData);
        showImage(selectMapData);
    }
});

let overlay;

function showImage(selectMapData) {
    const center = map.getCenter();
    const lon = center.lng;
    const lat = center.lat;
    const url = selectMapData.svg;
    const width = selectMapData.width;
    const height = selectMapData.height;
    const img = new Image();
    img.src = url;





    var point1 = L.latLng(lat, lon - 0.0006),
    point2 = L.latLng(lat, lon),
    point3 = L.latLng(lat - 0.0006, lon - 0.0006);

    var marker1 = L.marker(point1, { draggable: true }).addTo(map),
    marker2 = L.marker(point2, { draggable: true }).addTo(map),
    marker3 = L.marker(point3, { draggable: true }).addTo(map);
    

    var bounds = new L.LatLngBounds(point1, point2).extend(point3);

    map.fitBounds(bounds);



    if (overlay) {
        map.removeLayer(overlay);
    }

    overlay = L.imageOverlay.rotated(img, point1, point2, point3, {
        opacity: 0.4,
        interactive: true,
    });

    map.addLayer(overlay);


    function repositionImage() {
        const m1c = marker1.getLatLng();
        const m2c = marker2.getLatLng();
        const m3c = marker3.getLatLng();

        console.log("M1", m1c, "M2", m2c, "M3", m3c);
        overlay.reposition(m1c, m2c, m3c);

        document.getElementById("output").value = JSON.stringify([m1c, m2c, m3c]);

        // Convert coordinates to vectors
        const angle = getAngleToNorth(m1c.lat, m1c.lng, m2c.lat, m2c.lng);

        console.log(`Angle between perpendicular line and North Pole: ${angle.toFixed(2)} degrees`);
    };

    marker1.on('drag dragend', repositionImage);
    marker2.on('drag dragend', repositionImage);
    marker3.on('drag dragend', repositionImage);
}

function getAngleToNorth(lat1, lon1, lat2, lon2) {
    // Calculate the change in latitude and longitude
    const dLat = lat2 * Math.PI / 180 - lat1 * Math.PI / 180;
    const dLon = lon2 * Math.PI / 180 - lon1 * Math.PI / 180;
  
    // Calculate the angle (in radians)
    const angle = Math.atan2(-dLon, dLat);
  
    // Convert to degrees and adjust for north reference (0 degrees)
    const degrees = (angle * 180 / Math.PI + 90) % 360;
  
    return degrees;
  }
  