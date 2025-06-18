const MAP_DATA_JSON = "/static/SWMM_data.json";

let map = L.map('map', {
    attributionControl: true,
    zoomControl: true,
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
/*
var googleStreets = L.tileLayer('http://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',{
    maxZoom: 24,
    maxNativeZoom: 22,
    subdomains:['mt0','mt1','mt2','mt3']
}).addTo(map);
*/

protomapsL.leafletLayer({ url: 'https://api.protomaps.com/tiles/v3/{z}/{x}/{y}.mvt?key=35a84a0ee1532aef', theme: "light" }).addTo(map)

let transformer;


function refreshMap() {
    fetch(MAP_DATA_JSON)
        .then(response => response.json())
        .then(data => {
            console.log(data)
            updateMap(data);
            console.log("Map updated with new data.");
        });
}

//const refreshMap = document.getElementById("refresh-list");
//refreshButton.addEventListener('click', refreshMap);

refreshMap();

function updateMap(data) {

    // georeference map background image

    const m1cm = L.latLng(data.map.georeference_map[0]);
    const m2cm = L.latLng(data.map.georeference_map[1]);
    const m3cm = L.latLng(data.map.georeference_map[2]);
    const boundsm = new L.LatLngBounds(m1cm, m2cm).extend(m3cm);
    map.fitBounds(boundsm);

    let overlay = L.imageOverlay.rotated(data.map.svg, m1cm, m2cm, m3cm, {
        opacity: 0.95,
        interactive: true,
    });

    map.addLayer(overlay);

    // georeference tubes

    const m1c = L.latLng(data.map.georeference_tubes[0]);
    const m2c = L.latLng(data.map.georeference_tubes[1]);
    const m3c = L.latLng(data.map.georeference_tubes[2]);
    const bounds = new L.LatLngBounds(m1c, m2c).extend(m3c);
    map.fitBounds(bounds);

    const minLng = Math.min(m1c.lng, m2c.lng, m2c.lng)
    const minLat = Math.min(m1c.lat, m2c.lat, m2c.lat)

    const topLeft = [m1c.lng - minLng, m1c.lat - minLat]
    const topRight = [m2c.lng - minLng, m2c.lat - minLat]
    const bottomLeft = [m3c.lng - minLng, m3c.lat - minLat]
    const bottomRight = [topRight[0] - topLeft[0] + bottomLeft[0], topRight[1] - topLeft[1] + bottomLeft[1]]

    const srcCorners = [0, 0, data.map.width, 0, 0, data.map.height, data.map.width, data.map.height]
    const dstCorners = topLeft.concat(topRight).concat(bottomLeft).concat(bottomRight);

    transformer = PerspT(srcCorners, dstCorners);
    transformer.minLng = minLng;
    transformer.minLat = minLat;
    transformer.offset = [data.map.offset_x, data.map.offset_y];
    transformer.width = data.map.width;
    transformer.height = data.map.height;


    // add edges
    let num_overflowed = 0;

    data.edges.forEach(edge => {
    
        const color = edge.overflowed ? 'red' : '#235d97';

        let coords = [];
        edge.vertices.forEach(vertex => {
            const [x, y] = transformPoint(vertex.x, vertex.y);
            coords.push([y, x]);
        });

        const conduit = L.corridor(coords, {color: color, corridor: 0.3 + 5 * edge.depth}).addTo(map);

        conduit.bindTooltip(edge.name, {
            permanent: false,
            direction: 'top'
        });

        if (edge.overflowed) {
            num_overflowed++;
        }
    });

    // update gauge
    const overflowed_perc = num_overflowed / data.edges.length * 100
    gauge.set(overflowed_perc);
    document.getElementById("gauge_perc").innerText = overflowed_perc.toPrecision(2) + "%";


    // add nodes

    Object.entries(data.nodes).forEach(([key, node]) => {
        const [x, y] = transformPoint(node.x, node.y);
        let symbol;
        if (node.is_outfall) {
            symbol = drawTriangle([y, x], 20, { color: "black", opacity: 1, stroke: false, fillOpacity: 1 }).addTo(map);
        } else {
            symbol = L.circle([y, x], { color: "black", radius: 6, opacity: 1, stroke: false, fillOpacity: 1 }).addTo(map);
        }
        //L.tooltip().setLatLng([y, x]).setContent(key).addTo(map);
        symbol.bindTooltip(key, {
            permanent: false,
            direction: 'top'
        });
    });

}

function transformPoint(x, y) {
    if (transformer === undefined)
        return [x, y];
    
    let mx = x - transformer.offset[0];
    let my = transformer.height - (y - transformer.offset[1]);

    const dstPt = transformer.transform(mx, my);
    return [dstPt[0] + transformer.minLng, dstPt[1] + transformer.minLat]
}

function drawTriangle(centerPoint, sideLength, options) {
    var latLngSideLength = sideLength / 111000;  // 1 degree is approximately 111km

    // Calculate the height of the equilateral triangle
    var height = (Math.sqrt(3) / 2) * latLngSideLength;

    // Calculate the three points of the triangle
    var topLeft = [centerPoint[0] + height/3, centerPoint[1] - latLngSideLength/2];
    var topRight = [centerPoint[0] + height/3, centerPoint[1] + latLngSideLength/2];
    var bottom = [centerPoint[0] - 2*height/3, centerPoint[1]];

    // Create the polygon
    var triangle = L.polygon([
        topLeft,
        topRight,
        bottom
    ], options);

    return triangle;
}


// gauge

var opts = {
    lines: 12,
    angle: -0.11, // The span of the gauge arc
    lineWidth: 0.12, // The line thickness
    radiusScale: 0.8, // Relative radius
    pointer: {
      length: 0, // // Relative to gauge radius
      strokeWidth: 0, // The thickness
      color: '#000000' // Fill color
    },
    limitMax: false,     // If false, max value increases automatically if value > maxValue
    limitMin: false,     // If true, the min value of the gauge will be fixed
    generateGradient: true,
    highDpiSupport: true,     // High resolution support
    percentColors: [[0.0, "#a9d70b" ], [0.50, "#f9c802"], [1.0, "#ff0000"]], // !!!!
    strokeColor: '#E0E0E0',
    generateGradient: true
    
  };

var target = document.getElementById('gauge'); // your canvas element
var gauge = new Gauge(target).setOptions(opts); // create sexy gauge!
gauge.maxValue = 100; // set max gauge value
gauge.setMinValue(0);  // Prefer setter over gauge.minValue = 0
gauge.animationSpeed = 32; // set animation speed (32 is default value)
gauge.set(0); // set actual value