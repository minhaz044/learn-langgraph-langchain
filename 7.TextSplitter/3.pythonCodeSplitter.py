from langchain_text_splitters import PythonCodeTextSplitter

text = """
class Student:
    def __init__(self, name, age, student_id, department):
        self.name = name
        self.age = age
        self.student_id = student_id
        self.department = department
        self.marks = {}

    # Add marks for a subject
    def add_mark(self, subject, mark):
        self.marks[subject] = mark
        print(f"{subject} mark added successfully.")

    # Calculate average marks
    def calculate_average(self):
        if not self.marks:
            return 0

        return sum(self.marks.values()) / len(self.marks)

    # Determine grade
    def get_grade(self):
        average = self.calculate_average()

        if average >= 80:
            return "A+"
        elif average >= 70:
            return "A"
        elif average >= 60:
            return "B"
        elif average >= 50:
            return "C"
        elif average >= 40:
            return "D"
        else:
            return "F"

    # Check if student passed
    def is_passed(self):
        return self.calculate_average() >= 40

    # Display marks
    def display_marks(self):
        print("\n--- Marks ---")
        for subject, mark in self.marks.items():
            print(f"{subject}: {mark}")

    # Display complete student information
    def display_info(self):
        print("\n--- Student Information ---")
        print(f"Name       : {self.name}")
        print(f"Age        : {self.age}")
        print(f"Student ID : {self.student_id}")
        print(f"Department : {self.department}")
        print(f"Average    : {self.calculate_average():.2f}")
        print(f"Grade      : {self.get_grade()}")
        print(f"Status     : {'Passed' if self.is_passed() else 'Failed'}")

"""
splitter = PythonCodeTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

data = splitter.split_text(text)

for chunk in data:
    print(chunk)
    print("\n==================\n")
