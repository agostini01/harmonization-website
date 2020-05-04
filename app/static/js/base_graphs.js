
function getPairPlot() {
    var new_graph = ''
    // new_graph += '<div class="plot">'
    // new_graph += '<div class="title">'
    
    new_graph += `
    <div class="plot">
    <div class="title">`
    var d = new Date();
    d.format("hh:mm:ss tt");
    new_graph += '<h5>' + d + '</h5>'
    new_graph += '</div>'

    // Plotting the graph
    new_graph += '<div class="graph-img"><img src=/graphs/getplot/getpairplot/'

    // Features to be plotted
    // new_graph += arguments[0]

    new_graph += ' width="1000"></div>'
    new_graph += '</div>'

    document.getElementById("graphs-div").insertAdjacentHTML("afterbegin", new_graph)
}

// Submit post on submit
$('#get-form').on('submit', function (event) {
    event.preventDefault();
    console.log("form submitted!")  // sanity check
    create_post();
});

// AJAX for posting
function request_graph() {
    console.log("create post is working!") // sanity check
    console.log($('#post-text').val())
};
