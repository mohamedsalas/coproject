class Form(object):
    pass


if __name__ == "__main__":
    print('It works!')

from django import forms


class Form(forms.Form):
    chomStart = forms.FloatField(label='chomStart')
    chomEnd = forms.FloatField(label='chomEnd')
    pibStart = forms.FloatField(label='pibStart')
    pibEnd = forms.FloatField(label='pibEnd')
