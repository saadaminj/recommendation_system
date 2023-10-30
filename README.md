# recommendation_system
Recommendation system for a food based website. This repo contains model, synthetic data, and the demo website


Web Application:
•	As we go for the Python-Django combination for the web application development, where we follow the MVP model to develop the web application.
•	MVP stands for Model, View, pages.
•	Where data is stored in form of model, called by views and display by pages.
•	While talking about the UI, the whole front-End is combination of HTML, CSS, JavaScript, Ajax, and jQuery.
•	We also used the bootstrap for the better user experience.
•	As the Django provides the admin panel to administration of the web application. But in this we have created the merchant portal as well. Where merchant can list their products with its details. 
•	Merchant can have access about their own categories, subcategories, users, products. He/ she can add, edit, and delete the amongst.
•	You can have access of merchant portal by adding ‘/Merchant’ in the URL and you can get the Login page for the merchant portal.
•	 The Whole UI part can view in the user interface section.
•	Basically, the Recommendation will work on the model and the model needs to integrate with the application neatly and thoroughly.
•	So, for that we have used the pickle library to convert the model code in 
‘. pkl’ format. 
•	After converting that, we load that file into our view where we create the predict function which returns the list of recommended products after passing the user_id.
•	But the thing here is there is no direct login in the web application. So, that based on our previous recommendation the predict function can recommend the list of products while loading the Home or dashboard page of the application.
•	Now, if user login to the application, predict function gets the user_id as its parameter to pass to the model and the based on that user_id the model will recommend the product.
•	All synthetic Data generation part in Synthetic data section.
•	All models in Models Section
