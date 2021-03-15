
/**
 * Delete parent of the element.
 * @param  {document.element} e  The element whose parent will be deleted
 */
function deleteCard(e) {
    e.parentElement.remove();
}


/**
 * Insert a new card with the correct plot based on the django form.
 */

 
function getPlot() {

    var xmlHttp = new XMLHttpRequest();

    var URL = '/graphs/api/getinfo' 

    URL  +='?plot_type='+ document.getElementById("id_plot_type").value;
    URL  +='&x_feature='+ document.getElementById("id_x_feature").value;
    URL  +='&y_feature='+ document.getElementById("id_y_feature").value;
    URL  +='&color_by=' + document.getElementById("id_color_by").value;
    URL  +='&time_period=' + document.getElementById("id_time_period").value;
    URL  +='&fig_dpi='  + document.getElementById("id_fig_dpi").value;
    URL  +='&dataset_type='  + document.getElementById("id_dataset_type").value;
    URL  +='&include_covars='  + document.getElementById("id_include_covars").value;

    console.log(URL)
    xmlHttp.open( "GET", URL, false ); 
    xmlHttp.send( null );

    var summary_text = xmlHttp.responseText
 
    console.log("create get is working!")

    plot_name = document.getElementById("id_plot_name").value;
    console.log(plot_name)
    var d = new Date();
    d.format("hh:mm:ss tt");
    console.log(d)

    var new_graph = ''


    // https://coderthemes.com/hyper/modern/index.html
    new_graph+=`
    <div class="card">
        <div class="card-body">
            <div class="float-right"> 
                <i class="text-muted">`
     new_graph+=d+` </i>
            </div>
            <h5 class="text-muted font-weight-bold mt-0" title="Plot Title">`
     new_graph+=plot_name+`       
            </h5>`


    new_graph += `
    <div class="plot">
    <div class="title">`

    // Plotting the graph
    // This next line calls the django url that sends the request to the
    // API container
    new_graph += '<div class="graph-img"><img src=/graphs/api/getplot'

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
    new_graph +='&time_period=' + document.getElementById("id_time_period").value;
    new_graph +='&fig_dpi='  + document.getElementById("id_fig_dpi").value;
    new_graph +='&dataset_type='  + document.getElementById("id_dataset_type").value;
    new_graph +='&include_covars='  + document.getElementById("id_include_covars").value;
    new_graph += ' style="max-width:100%; height:auto"></div>'
    new_graph += `</div>
        </div > `
        + summary_text + `
        <button type="button" class="btn btn-outline-secondary btn-sm" onclick="deleteCard(this)">Remove Graph</button>
    </div >
    `

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
