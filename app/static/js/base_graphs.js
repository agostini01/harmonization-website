
function getPairPlot() {
    var new_graph = ''
    new_graph += '<div class="plot">'
    new_graph += '<div class="title">'

    var d = new Date();
    d.format("hh:mm:ss tt");

    new_graph += '<h5>'+d+'</h5>'
    new_graph += '</div>'
    new_graph +=
         '<div class="graph-img"><img src=/graphs/getplot/getpairplot/ width="1000"></div>'
    new_graph += '</div>'
    document.getElementById("graphs-div").insertAdjacentHTML("afterbegin", new_graph)
}