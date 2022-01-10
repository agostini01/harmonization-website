/* Insert a new card with the correct plot based on the django form.
 */

 
function getPlot() {

    var xmlHttp = new XMLHttpRequest();

    var URL = '/graphs/api/getinfo' 

    URL  +='?plot_type='+ document.getElementById("id_search_field").value;
    

    /**
     * Selection for specific covariates
     */

    console.log(URL)
    xmlHttp.open( "GET", URL, false ); 
    xmlHttp.send( null );

    var summary_text = xmlHttp.responseText
 
    console.log("create get is working!")

    summary_text+='<div>'
        + summary_text + 
    '</div >'
    

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
