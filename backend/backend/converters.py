class BooleanConverter:
    regex = 'true|false'

    def to_python(self, value):
        return value.lower() == 'true'

    def to_url(self, value):
        return str(value).lower()