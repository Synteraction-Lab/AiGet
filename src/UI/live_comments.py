import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

COMMENT_BOX_WIDTH = 1000


class LiveCommentsApp:
    def __init__(self, root):
        self.root = root
        self.input_list = []
        self.screen_height = self.root.winfo_screenheight()
        self.comments_area_height = self.screen_height // 10

        self.comments_frame = ctk.CTkFrame(self.root, height=self.comments_area_height, width=COMMENT_BOX_WIDTH,
                                           fg_color='black', bg_color='black')
        self.comments_frame.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)

        self.canvas = ctk.CTkCanvas(self.comments_frame, bg='black', height=self.comments_area_height,
                                    width=COMMENT_BOX_WIDTH - 5, bd=0, highlightthickness=0, relief='ridge')
        self.scrollbar = ctk.CTkScrollbar(self.comments_frame, orientation="vertical", command=self.canvas.yview,
                                          bg_color='black')
        self.scrollable_frame = ctk.CTkFrame(self.canvas, bg_color='black', fg_color='black')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.comments = []

        self.root.after(1000, self._add_comment)

        # Bind mouse wheel to scroll
        self.root.bind_all("<MouseWheel>", self._on_mouse_wheel)

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1 * event.delta), "units")

    def add_new_item(self, items):
        self.input_list.extend(items)

    def _add_comment(self):
        time_gap = 4000
        if self.input_list:
            text = self.input_list.pop(0)
            self.create_comment(text)
            time_gap = min(max(4000, len(text.split()) * 400), 8000)
        self.root.after(time_gap, self._add_comment)

    def create_comment(self, text):
        comment_label = tk.Label(self.scrollable_frame, text=text, background="black", foreground='#19FF53',
                                 font=('Robot', 32), anchor='w', wraplength=COMMENT_BOX_WIDTH - 5, justify='left')
        comment_label.pack(fill=tk.X, pady=5)
        comment_label.bind("<Button-1>", self._on_comment_click)
        self.comments.append(comment_label)
        self.update_scroll_region()

    def _on_comment_click(self, event):
        comment_text = event.widget.cget("text")
        print(f"Clicked on comment: {comment_text}")

    def update_scroll_region(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1)

    def pack_forget(self):
        self.comments_frame.pack_forget()

    def pack(self):
        self.comments_frame.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)

    def winfo_ismapped(self):
        return self.comments_frame.winfo_ismapped()

    def run(self):
        self.root.mainloop()


def generate_live_comments():
    # Sample suggestions based on the given real cases, concise and in mixed Chinese-English style
    suggestions = ['Did you know that instant noodles were invented by Momofuku Ando in 1958?',
                   'Instant noodles are often fortified with additional vitamins and minerals to boost their nutritional value.',
                   'There are over 100 billion servings of instant noodles consumed worldwide annually.',
                   "The first flavor of instant noodles was chicken, known as 'Chikin Ramen'.",
                   'The world’s most expensive instant noodles cost over $50 per pack.',
                   'Instant noodles were originally considered a luxury item in Japan.',
                   'Cup Noodles were introduced in 1971 by Nissin.',
                   'There are instant noodles that can be prepared by adding cold water.',
                   'South Korea has the highest per capita consumption of instant noodles.',
                   'In Japan, there are instant noodles vending machines.',
                   'The longest instant noodle was 3,084 meters long, made in China in 2017.',
                   'Some instant noodles are specifically made for space travel.',
                   'Instant noodles are popular in prisons due to their affordability and ease of preparation.',
                   'In Indonesia, instant noodles are often served with rice.',
                   'In some cultures, people add peanut butter to their instant noodles for extra flavor.']

    return suggestions


if __name__ == "__main__":
    input_list = generate_live_comments()

    root = tk.Tk()
    toplevel = tk.Toplevel(background='black')
    # overdirect is used to make the window appear on top of all other windows
    # toplevel.overrideredirect(True)
    toplevel.attributes('-topmost', True)
    app = LiveCommentsApp(toplevel)
    app.add_new_item(input_list)
    app.run()
