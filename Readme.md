# Churn Data Analisys

The code contained in this repository is related with the data analysis of sellers performance and the company churn data.

The initial idea was to build a model to identify trends, and then to correlate the trends with a churn event, to eventually check if the initial hypothesis, that growing trends or declining trends can lead to a churn event.

## Getting Started

First of all, we load the .csv files as pandas dataframes.

```python
seller_per = pd.read_csv('seller_performance_data.csv', sep=';', decimal=",")
churn_data = pd.read_csv('churn_data.csv', sep=';', decimal=",")
```
We can check then the head of both dataframes

for the seller_per we have
 
  | Unnamed: 0 |                         supplier_key  | report_date  |    ... |       units_ordered | units_ordered_b2b | units_refunded
  | -----------|---------------------------------------|--------------|--------|---------------------|-------------------|----------------
0 |          1 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-24   |    ... |                 214 |                 0 |              0
1 |          2 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-23   |    ... |                 193 |                 0 |              0
2 |          3 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-22   |    ... |                 188 |                 0 |              0
3 |          4 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-21   |    ... |                 160 |                 0 |              0
4 |          5 | 00179d2b-e696-4536-b530-e25ed838fae6  | 2018-10-20   |    ... |                  86 |                 0 |              0

and for churn_data we have

   Unnamed: 0                          supplier_key  churn_date
0           1  031a13f1-5488-4ef8-a2fa-e55bb894c44e  2018-01-18
1           2  03d96d8a-7178-4f7d-a8f2-8ef1a643ecd5  2016-12-09
2           3  0574ad4e-5331-4e0b-9dd5-70d8fd3ba01b  2018-01-19
3           4  073941ba-16e0-4a1f-825e-1cd94b9d50cb  2017-04-27
4           5  082ba7d5-7b68-46e3-8a53-fc4d52104e28  2018-10-02

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

