// var script = document.createElement('script');
// script.src = 'http://code.jquery.com/jquery-1.11.0.min.js';
// script.type = 'text/javascript';
// document.getElementsByTagName('head')[0].appendChild(script);
var search = "";

const app = document.getElementById('root');
const logo = document.createElement('img');
logo.src = '';

const container = document.createElement('div');
container.setAttribute('class', 'container');

//app.appendChild(logo);
app.appendChild(container);

var request = new XMLHttpRequest();
request.open('POST', 'http://127.0.0.1:8000/getAll/', true);
request.send('travel');
request.onload = function () {
var data = JSON.parse(this.response);
console.log('google-request', data);
if (request.status >= 200 && request.status < 400)
{
    var j;
    hlist = ["Google-results"]
    //hlist = ["Google-results","Bing-results","Our-results"]
    for(j=0;j<hlist.length;j++)
    {
      const card = document.createElement('div')
      card.setAttribute('class', 'card');
      card.setAttribute('id', hlist[j]);

      const h1 = document.createElement('h1');
      h1.textContent = hlist[j];
      var i;
      text ="<ul>";

      list = data[hlist[j]];
      for (i = 0; i < list.length; i++) {
        text +="<a href ="+list[i]+"><li>"+list[i]+"</li></a>";
      }
      text += "</ul>";

      const p = document.createElement('p');
      p.textContent = 'cluster id';

      container.appendChild(card);
      card.appendChild(h1);
      $('#'+hlist[j]).append(text);
      card.appendChild(p);}
}else
{
      const errorMessage = document.createElement('marquee');
      errorMessage.textContent = `Gah, it's not working!`;
      app.appendChild(errorMessage);
}
}
