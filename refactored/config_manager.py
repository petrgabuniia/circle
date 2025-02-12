import configparser
import tkinter as tk


config = configparser.ConfigParser()

def load_config():
    if not config.read('data.ini'):
        create_config_section('Info')
        config.set('Info', 'operator_name', "")
        config.set('Info', 'standard_name', "")
        config.set('Info', 'standard_id', "")
        config.set('Info', 'unit_id', "")
        save_config()
    return {
        'operator_name': config.get('Info', 'operator_name', fallback=""),
        'standard_name': config.get('Info', 'standard_name', fallback=""),
        'standard_id': config.get('Info', 'standard_id', fallback=""),
        'unit_id': config.get('Info', 'unit_id', fallback="")
    }

def save_config():
    try:
        with open('data.ini', 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        tk.messagebox.showerror("Error", f"Could not save the configuration: {e}")

def create_config_section(section):
    if not config.has_section(section):
        config.add_section(section)