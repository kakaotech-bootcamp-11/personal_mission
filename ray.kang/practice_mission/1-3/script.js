document.addEventListener("DOMContentLoaded", () => {
  const todoInput = document.getElementById("todoInput");
  const addTodoButton = document.getElementById("addTodo");
  const todoList = document.getElementById("todoList");

  if (!todoList) throw new Error("todoList element not found");
  if (!addTodoButton) throw new Error("addTodo button not found");
  if (!todoInput) throw new Error("todoInput element not found");

  // 로컬 스토리지에서 Todo 목록 불러오기
  const todosString = localStorage.getItem("todos");
  let todos = todosString ? JSON.parse(todosString) : [];

  // Todo 목록 렌더링 함수
  function renderTodos() {
    if (!todoList) return;

    todoList.innerHTML = "";
    todos.forEach((todo, index) => {
      const li = document.createElement("li");
      li.className = "todo-item";
      li.innerHTML = `
      <input type="checkbox" ${todo.completed ? "checked" : ""}>
      <span class="${todo.completed ? "completed" : ""}">${todo.text}</span>
      <i class="fas fa-times delete-icon"></i>
    `;

      // 체크박스 이벤트
      const checkbox = li.querySelector('input[type="checkbox"]');
      if (checkbox) {
        checkbox.addEventListener("change", () => toggleTodo(index));
      }

      // 삭제 버튼 이벤트
      const deleteIcon = li.querySelector(".delete-icon");
      if (deleteIcon) {
        deleteIcon.addEventListener("click", (e) => {
          e.stopPropagation();
          deleteTodo(index);
        });
      }

      todoList.appendChild(li);
    });
    saveTodos();
  }

  // 새 Todo 추가 함수
  function addTodo() {
    if (!todoInput || !(todoInput instanceof HTMLInputElement)) return;

    const todoText = todoInput.value.trim();
    if (todoText) {
      todos.push({ text: todoText, completed: false });
      todoInput.value = "";
      renderTodos();
    }
  }

  // Todo 토글 함수
  function toggleTodo(index) {
    todos[index].completed = !todos[index].completed;
    renderTodos();
  }

  // Todo 삭제 함수
  function deleteTodo(index) {
    todos.splice(index, 1);
    renderTodos();
  }

  // 로컬 스토리지에 Todo 목록 저장
  function saveTodos() {
    localStorage.setItem("todos", JSON.stringify(todos));
  }

  // 이벤트 리스너
  addTodoButton.addEventListener("click", addTodo);
  todoInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") addTodo();
  });

  // 초기 렌더링
  renderTodos();
});
