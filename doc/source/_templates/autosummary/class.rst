{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :no-members:
    :no-inherited-members:
    :no-special-members:

    {% block methods %}
    ..
        .. autosummary::
            :toctree:
        {% for item in all_methods %}
            {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
            {{ name }}.{{ item }}
            {%- endif -%}
        {%- endfor %}
        {% for item in inherited_members %}
            {%- if item in ['__call__', '__mul__', '__getitem__', '__len__', '__pow__'] %}
            {{ name }}.{{ item }}
            {%- endif -%}
        {%- endfor %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    ..
        .. autosummary::
            :toctree:
        {% for item in all_attributes %}
            {%- if not item.startswith('_') %}
            {{ name }}.{{ item }}
            {%- endif -%}
        {%- endfor %}
    {% endif %}
    {% endblock %}
