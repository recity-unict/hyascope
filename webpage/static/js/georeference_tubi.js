const MAP_DATA_JSON = "static/data.json";

let map = L.map('map', {
    attributionControl: false,
    zoomControl: false,
    minZoom: -4,
}).setView([37.52561664048445, 15.074417095820316], 15);


let transformer;
let marker1, marker2, marker3;
let data;

protomapsL.leafletLayer({ url: 'https://api.protomaps.com/tiles/v3/{z}/{x}/{y}.mvt?key=35a84a0ee1532aef', theme: "light" }).addTo(map)


fetch(MAP_DATA_JSON)
    .then(response => response.json())
    .then(d => {
        console.log(d)
        data = d;
        showTubes();
        redrawTubes();
        console.log("Map updated with new data.");
    });


function showTubes() {
    if (data === undefined) return;

    const center = map.getCenter();
    const lon = center.lng;
    const lat = center.lat;

    var m1c = L.latLng(lat, lon - 0.0006);
    var m2c = L.latLng(lat, lon);
    var m3c = L.latLng(lat - 0.0006, lon - 0.0006);

    marker1 = L.marker(m1c, { draggable: true }).addTo(map);
    marker2 = L.marker(m2c, { draggable: true }).addTo(map);
    marker3 = L.marker(m3c, { draggable: true }).addTo(map);

    marker1.on('drag dragend', redrawTubes);
    marker2.on('drag dragend', redrawTubes);
    marker3.on('drag dragend', redrawTubes);

    calculateTransformer();
}

function calculateTransformer() 
{
    const minLng = Math.min(marker1.getLatLng().lng, marker2.getLatLng().lng, marker3.getLatLng().lng)
    const minLat = Math.min(marker1.getLatLng().lat, marker2.getLatLng().lat, marker3.getLatLng().lat)

    const topLeft = [marker1.getLatLng().lng - minLng, marker1.getLatLng().lat - minLat]
    const topRight = [marker2.getLatLng().lng - minLng, marker2.getLatLng().lat - minLat]
    const bottomLeft = [marker3.getLatLng().lng - minLng, marker3.getLatLng().lat - minLat]

    const bottomRight = [topRight[0] - topLeft[0] + bottomLeft[0], topRight[1] - topLeft[1] + bottomLeft[1]]

    const srcCorners = [0, 0, data.map.width, 0, 0, data.map.height, data.map.width, data.map.height]
    const dstCorners = topLeft.concat(topRight).concat(bottomLeft).concat(bottomRight);

    transformer = PerspT(srcCorners, dstCorners);
    transformer.minLng = minLng;
    transformer.minLat = minLat;
    transformer.offset = [data.map.offset_x, data.map.offset_y];
    transformer.width = data.map.width;
    transformer.height = data.map.height;

    document.getElementById("output").value = JSON.stringify([marker1.getLatLng(), marker2.getLatLng(), marker3.getLatLng()]);

}


let edges = [];
let nodes = [];


function redrawTubes() {
    if (data === undefined) return;

    for (let i = 0; i < edges.length; i++) {
        edges[i].remove();
    }
    edges = [];

    for (let i = 0; i < nodes.length; i++) {
        nodes[i].remove();
    }
    nodes = [];

    calculateTransformer();

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

        edges.push(conduit);
    });


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

        nodes.push(symbol);
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