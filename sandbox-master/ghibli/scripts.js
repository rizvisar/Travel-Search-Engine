// var script = document.createElement('script');
// script.src = 'http://code.jquery.com/jquery-1.11.0.min.js';
// script.type = 'text/javascript';
// document.getElementsByTagName('head')[0].appendChild(script);

function setCharAt(str,index,chr) {
    if(index > str.length-1) return str;
    return str.substr(0,index) + chr + str.substr(index+1);
}

window.onload = function() {


  $("#searchBtn").on('click',function(){
  var search = $("#search").val();
  const app = document.getElementById('root');
  const logo = document.createElement('img');
  logo.src = '';

  const container = document.createElement('div');
  container.setAttribute('class', 'container');

  app.appendChild(logo);
  app.appendChild(container);

  var request = new XMLHttpRequest();
  request.open('POST', 'http://127.0.0.1:8000/getAll/', true);
  request.onload = function () {
      var data = JSON.parse(this.response);
      console.log('google-request', data);
      if (request.status >= 200 && request.status < 400)
      {
        var j;
      //hlist = ["results"]HubScore-Results':hubBasedResult,'Auth-Results':
      hlist = ["Google-results","Bing-results","Our-results","Our-PageRank-results","Hub-Results","Auth-Results","Metric-Results","Rocchio-Results","Kmeans-Results","Agglomerative-Results"]

        //
        for(j=0;j<hlist.length;j++)
        {
          const card = document.createElement('div')
          card.setAttribute('class', 'card');
          card.setAttribute('id', hlist[j]);

          var expandedQuery = "";
          if(hlist[j] === "Metric-Results")
          {
            expandedQuery = data["Query_Metric"];
          }
          if(hlist[j] === "Rocchio-Results")
          {
            expandedQuery = data["Query_Rocchio"];
          }
          const h1 = document.createElement('h1');
          h1.textContent = hlist[j];
          var i;
          text = expandedQuery !== "" ? "<p>Expanded Query : "+ expandedQuery+"</p>" : "";
          text +="<ul>";
          list = data[hlist[j]];
          for (i = 0; i < list.length; i++) {
            var opurl = list[i];
            if(hlist[j] === "Kmeans-Results"){
              clist = data["KmeansCluster"]
              text +="<li><a href ="+opurl+">"+opurl+"</a> {CLuster ID} : "+clist[i]+"</li>"
            }
            if(hlist[j] === "Agglomerative-Results"){
              alist = data["AggCluster"]
              text +="<li><a href ="+opurl+">"+opurl+"</a> {CLuster ID} : "+alist[i]+"</li>"
            }
            else{
            text +="<li><a href ="+opurl+">"+opurl+"</a></li>"
          }
            }
          text += "</ul>";

          container.appendChild(card);
          card.appendChild(h1);
          $('#'+hlist[j]).append(text);
          }

      }else
      {
        const errorMessage = document.createElement('marquee');
        errorMessage.textContent = `Gah, it's not working!`;
        app.appendChild(errorMessage);
      }
    }
    request.send(search);

  });
}
