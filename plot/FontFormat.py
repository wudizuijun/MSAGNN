class FontFormat():
    def __init__(self) -> None:
        # set the font format
        self.label_font = {'family': 'Times New Roman',
            'size': 12,
            'weight': 'normal',
        }
        self.title_font = {'family': 'Times New Roman',
            'size': 13,
            'weight': 'bold',
        }

        self.ticks_font = {'family': 'Times New Roman',
            'size': 12,
            'weight': 'normal',
        }
        self.legend_font = {'family': 'Times New Roman',
            'size': 8,
            'weight': 'bold',
        }
    
    def set_label_font(self, size=12, bold=False, family='Times New Roman'):
        self.label_font['size'] = size
        self.label_font['weight'] = 'bold' if bold else 'normal'
        self.label_font['family'] = family
            
    def set_title_font(self, size=12, bold=False, family='Times New Roman'):
        self.title_font['size'] = size
        self.title_font['weight'] = 'bold' if bold else 'normal'
        self.title_font['family'] = family
        
    def set_ticks_font(self, size=12, bold=False, family='Times New Roman'):
        self.ticks_font['size'] = size
        self.ticks_font['weight'] = 'bold' if bold else 'normal'
        self.ticks_font['family'] = family
        
    def set_legend_font(self, size=12, bold=False, family='Times New Roman'):
        self.legend_font['size'] = size
        self.legend_font['weight'] = 'bold' if bold else 'normal'
        
    @property
    def get_label_font(self):
        return self.label_font
    
    @property
    def get_title_font(self):
        return self.title_font
    
    @property
    def get_ticks_font(self):
        return self.ticks_font
    
    @property
    def get_legend_font(self):
        return self.legend_font