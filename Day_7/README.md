# Day 7
## Web Deployment using Flask
> I need to break away from core ML for a while because I need to implement/showcase an analytics project I did some back as a web app for my college presentation. Nevertheless, I'll keep logging in the gained knowledge here!

### **Flask** is a microframework in python which is helpful in deploying web apps and writing frontend in Python  
* Other Alternatives include Django and Angular.js
* It is used in integration with HTML and CSS which provides easy structure and style to web pages.
* Create a folder 'templates' in your project folder and add all the html and css files in it. These can be access by using the module `render_template` from flask. 
* Flask acts as an intermediary between frontend user input/navigation and the respective HTML webpages responses.
* **Bootstrap** is another technique used by programmers to use pre-defined templates of certain structure and style for quick fixes in the web-apps. The documentation on [get.bootstrap.com](https:\\get.bootstrap.com) is a good place to start experimenting with HTML tags. 

`pip install flask`

* After working around on the flask app and getting the hang of it I was able to create a good looking header on various different webpages  
![Flask app](../Capture.JPG)

I have to now insert my charts, graphs and maps. There are many options do to this:
- save the graphs as .png files in a static folder and put it in HTML directly; static & non-interactive
- compute graph inside flask and pass it to HTML directly; dynamic but non-interactive
- use exernal library to create graphs inside flask; dynamic and interactive

### **Plotly Dash** is a library which uses flask under the hood and is useful for making charts/graphs on web-pages
*I have two options at the moment* 
1) to integrate flask with plotly as a tool for the graphs 
2) use plotly completely to design the app without using flask  
I'll try out both of them, compare and see which one is better for myself

## Using Dash Plotly for visualisation 


