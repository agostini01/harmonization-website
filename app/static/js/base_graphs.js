
function getPlot() {
    var new_graph = ''
    // new_graph += '<div class="plot">'
    // new_graph += '<div class="title">'
   
    // TODO: Add discard buttom
    new_graph += `
    <div class="plot">
    <div class="title">`
    var d = new Date();
    // TODO: Title too big
    d.format("hh:mm:ss tt");
    new_graph += '<h5>' + d + '</h5>'
    new_graph += '</div>'

    // Plotting the graph
    new_graph += '<div class="graph-img"><img src=/graphs/getplot'

    // Features to be plotted
    // new_graph += arguments[0]
    // for (var a of arguments) {
    //     console.log(a);
    //     new_graph +='&x_feature'+a
    // }

    new_graph +='?plot_type='+ document.getElementById("id_plot_type").value;
    new_graph +='&x_feature='+ document.getElementById("id_x_feature").value;
    new_graph +='&y_feature='+ document.getElementById("id_y_feature").value;
    new_graph +='&color_by=' + document.getElementById("id_color_by").value;
    // TODO: Plot title
    new_graph += ' width="1000"></div>'
    new_graph += '</div>'

    document.getElementById("graphs-div").insertAdjacentHTML("afterbegin", new_graph)
}

// Submit post on submit
// $('#get-form').on('submit', function (event) {
//     event.preventDefault();
//     console.log("form submitted!")  // sanity check
//     create_post();
// });

// AJAX for posting
function request_graph() {
    console.log("create post is working!") // sanity check
    console.log($('#post-text').val())
};
