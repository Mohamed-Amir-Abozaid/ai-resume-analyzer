const API_URL = "http://127.0.0.1:8000";

const authShell = document.querySelector("#auth-shell");
const appShell = document.querySelector("#app-shell");

const authForm = document.querySelector("#auth-form");
const nameField = document.querySelector("#name-field");
const termsRow = document.querySelector("#terms-row");

const authTitle = document.querySelector("#auth-title");
const authSubtitle = document.querySelector("#auth-subtitle");
const authSubmitLabel = document.querySelector("#auth-submit-label");
const authFootnote = document.querySelector("#auth-footnote");

const passwordInput = document.querySelector("#password");
const resumeInput = document.querySelector("#resume-input");

const sidebar = document.querySelector("#sidebar");

const modal = document.querySelector("#modal-backdrop");
const jobModal = document.querySelector("#job-modal-backdrop");
const jobForm = document.querySelector("#job-form");

const toast = document.querySelector("#toast");
const toastMessage = document.querySelector("#toast-message");

const views = [...document.querySelectorAll(".view")];
const navItems = [...document.querySelectorAll(".nav-item")];

let authMode = "register";
let toastTimer;


const titles = {
  overview: "Overview",
  analysis: "Resume analysis",
  resumes: "My resumes",
  jobs: "Jobs",
  "job-detail": "Job match",
  improvement: "Improve resume"
};


/* =========================
   TOAST
========================= */

function showToast(message) {

  toastMessage.textContent = message;

  toast.classList.add("show");

  clearTimeout(toastTimer);

  toastTimer = setTimeout(() => {
    toast.classList.remove("show");
  }, 3400);
}


function getApiError(data, fallback) {

  return data?.detail || data?.error || data?.message || fallback;
}


/* =========================
   AUTH MODE
========================= */

function setAuthMode(mode) {

  authMode = mode;

  const isRegister = mode === "register";

  nameField.hidden = !isRegister;
  termsRow.hidden = !isRegister;

  nameField.querySelector("input").required = isRegister;
  termsRow.querySelector("input").required = isRegister;

  authTitle.textContent =
    isRegister
      ? "Create your account"
      : "Welcome back";

  authSubtitle.textContent =
    isRegister
      ? "Your personalized career workspace starts here."
      : "Log in to continue building your next career move.";

  authSubmitLabel.textContent =
    isRegister
      ? "Create account"
      : "Log in";

  authFootnote.innerHTML =
    isRegister
      ? `Already registered?
         <button class="text-button" data-auth-mode="login">
           Log in instead
         </button>`
      : `New to ResumePilot?
         <button class="text-button" data-auth-mode="register">
           Create an account
         </button>`;

  document.querySelectorAll(".auth-tab").forEach((tab) => {

    tab.classList.toggle(
      "active",
      tab.dataset.authMode === mode
    );

  });

  document.querySelector(".auth-switch").textContent =
    isRegister
      ? "Log in"
      : "Register";

  passwordInput.autocomplete =
    isRegister
      ? "new-password"
      : "current-password";
}


/* =========================
   SHOW APP
========================= */

function showApp(userName) {

  const displayName =
    userName.trim() || "there";

  const firstName =
    displayName.split(/\s+/)[0];

  authShell.classList.add("hidden");

  appShell.classList.remove("hidden");

  document.querySelector("#welcome-name").textContent =
    firstName;

  document.querySelector("#profile-name").textContent =
    displayName;

  document.querySelector("#profile-avatar").textContent =
    displayName
      .split(/\s+/)
      .slice(0, 2)
      .map((part) => part[0])
      .join("")
      .toUpperCase();

  setView("overview");
}


/* =========================
   NAVIGATION
========================= */

function setView(viewName) {

  const nextView =
    titles[viewName]
      ? viewName
      : "overview";

  views.forEach((view) => {

    view.classList.toggle(
      "active-view",
      view.dataset.section === nextView
    );

  });

  navItems.forEach((item) => {

    item.classList.toggle(
      "active",
      item.dataset.view === nextView
    );

  });

  document.querySelector("#breadcrumb-title").textContent =
    titles[nextView];

  window.history.replaceState(
    null,
    "",
    `#${nextView}`
  );


  /*
   * Load jobs whenever
   * the Jobs page is opened.
   */

  if (nextView === "jobs") {
    loadJobs();
  }

  if (nextView === "resumes") {
    loadResumes();
  }

  if (nextView === "improvement") {
    loadCareerAdvice();
  }

  sidebar.classList.remove("open");
}


/* =========================
   CAREER ADVISOR MODAL
========================= */

async function loadCareerAdvice() {

  const loading = document.querySelector("#career-advisor-loading");
  const empty = document.querySelector("#career-advisor-empty");
  const result = document.querySelector("#career-advisor-result");
  const resumeId = localStorage.getItem("resume_id");

  loading.hidden = false;
  empty.hidden = true;
  result.hidden = true;

  if (!resumeId) {
    loading.hidden = true;
    empty.hidden = false;
    return;
  }

  try {
    const response = await fetch(
      `${API_URL}/resume/${resumeId}/career-advice`,
      { method: "POST" }
    );
    const data = await response.json();

    if (!response.ok || data.error) {
      throw new Error(getApiError(data, "Could not load career advice."));
    }

    document.querySelector("#career-advice-text").textContent =
      data.career_advice || "No career advice was generated yet.";

    renderList(
      "#career-missing-skills",
      data.missing_skills
    );

    renderAdvisorList(
      "#career-roadmap",
      data.learning_roadmap
    );

    renderAdvisorList(
      "#career-certifications",
      data.certifications
    );

    renderAdvisorList(
      "#career-resume-improvements",
      data.resume_improvements
    );

    loading.hidden = true;
    result.hidden = false;
  } catch (error) {
    console.error(error);
    loading.hidden = true;
    empty.hidden = false;
    empty.querySelector("h2").textContent = "Career advice unavailable";
    empty.querySelector("p").textContent = error.message;
  }
}

function openAdvisor() {

  modal.hidden = false;

  const result = document.querySelector("#advisor-result");
  const button = modal.querySelector("[data-action='send-advisor']");

  result.hidden = true;
  result.textContent = "";
  button.disabled = false;
  button.textContent = "Get advice →";

  modal
    .querySelector("textarea")
    .focus();
}


function closeAdvisor() {

  modal.hidden = true;
}


function openJobModal() {

  jobModal.hidden = false;
  jobForm.reset();
  delete jobForm.dataset.editingId;
  document.querySelector("#job-modal-title").textContent = "Add a job opportunity";
  jobModal.querySelector("input")?.focus();
}


function closeJobModal() {

  jobModal.hidden = true;
}


async function openEditJobModal(jobId) {

  try {
    const response = await fetch(`${API_URL}/jobs/${jobId}`);
    const job = await response.json();

    if (!response.ok || job.error) {
      showToast(getApiError(job, "Could not load job."));
      return;
    }

    jobForm.dataset.editingId = jobId;
    jobForm.elements.title.value = job.title || "";
    jobForm.elements.company.value = job.company || "";
    jobForm.elements.location.value = job.location || "";
    jobForm.elements.salary.value = job.salary || "";
    jobForm.elements.job_type.value = job.job_type || "Full-time";
    jobForm.elements.description.value = job.description || "";
    jobForm.elements.required_skills.value = Array.isArray(job.required_skills)
      ? job.required_skills.join(", ")
      : "";

    document.querySelector("#job-modal-title").textContent = "Edit job opportunity";
    jobModal.hidden = false;
    jobForm.elements.title.focus();
  } catch (error) {
    console.error(error);
    showToast("Could not connect to the server.");
  }
}


async function deleteJob(jobId) {

  if (!window.confirm("Delete this job and its saved matches?")) {
    return;
  }

  try {
    const response = await fetch(`${API_URL}/jobs/${jobId}`, {
      method: "DELETE"
    });
    const data = await response.json();

    if (!response.ok || data.error) {
      showToast(getApiError(data, "Could not delete job."));
      return;
    }

    showToast("Job deleted successfully.");
    loadJobs();
  } catch (error) {
    console.error(error);
    showToast("Could not connect to the server.");
  }
}


async function requestCareerAdvice() {

  const resumeId = localStorage.getItem("resume_id");
  const result = document.querySelector("#advisor-result");
  const button = modal.querySelector("[data-action='send-advisor']");

  if (!resumeId) {
    showToast("Upload and analyze your resume first.");
    return;
  }

  button.disabled = true;
  button.textContent = "Preparing your advice...";
  result.hidden = false;
  result.textContent = "Reading your resume and match history...";

  try {
    const response = await fetch(
      `${API_URL}/resume/${resumeId}/career-advice`,
      { method: "POST" }
    );
    const data = await response.json();

    if (!response.ok || data.error) {
      result.textContent = getApiError(data, "Could not prepare career advice.");
      return;
    }

    result.textContent = data.career_advice || "No advice was generated yet.";
  } catch (error) {
    console.error(error);
    result.textContent = "Could not connect to the server.";
  } finally {
    button.disabled = false;
    button.textContent = "Get advice →";
  }
}


/* =========================
   RESUME UPLOAD
========================= */

async function handleUpload(file) {

  if (!file) {
    return;
  }


  const validMimeTypes = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ];


  /*
   * Validate extension and MIME type.
   */

  if (
    !/\.(pdf|docx)$/i.test(file.name) &&
    !validMimeTypes.includes(file.type)
  ) {

    showToast(
      "Please choose a PDF or DOCX file."
    );

    return;
  }


  /*
   * Maximum file size = 10 MB.
   */

  if (file.size > 10 * 1024 * 1024) {

    showToast(
      "That file is larger than 10 MB."
    );

    return;
  }


  /*
   * Get logged-in user.
   */

  let user;

  try {
    user = JSON.parse(localStorage.getItem("user"));
  } catch (error) {
    localStorage.removeItem("user");
  }


  if (!user) {

    showToast(
      "Please log in first."
    );

    return;
  }


  /*
   * Create multipart form data.
   */

  const formData = new FormData();

  formData.append(
    "user_id",
    user.id
  );

  formData.append(
    "file",
    file
  );


  showToast(
    "Uploading and analyzing your resume..."
  );


  try {

    const response = await fetch(
      `${API_URL}/resume/upload`,
      {
        method: "POST",
        body: formData
      }
    );


    const data = await response.json();


    if (!response.ok || data.error) {

      showToast(
        getApiError(data, "Resume analysis failed.")
      );

      return;
    }


    console.log(
      "Resume analysis:",
      data
    );


    /*
     * Save resume ID for later
     * job matching and career advice.
     */

    localStorage.setItem(
      "resume_id",
      data.resume_id
    );


    showToast(
      "Resume analyzed successfully."
    );


    setView("analysis");


    displayResumeAnalysis(
      data.analysis
    );


  } catch (error) {

    console.error(error);

    showToast(
      "Could not connect to the server."
    );
  }
}


/* =========================
   DISPLAY RESUME ANALYSIS
========================= */

function displayResumeAnalysis(analysis) {

  const result =
    document.querySelector("#analysis-result");

  const emptyState =
    document.querySelector("#analysis-empty");


  result.hidden = false;

  emptyState.style.display = "none";


  document.querySelector(
    "#resume-summary"
  ).textContent =
    analysis.summary ||
    "No summary available.";


  renderList(
    "#technical-skills",
    analysis.technical_skills
  );


  renderList(
    "#soft-skills",
    analysis.soft_skills
  );


  renderList(
    "#education",
    analysis.education
  );


  renderList(
    "#experience",
    analysis.experience
  );


  renderList(
    "#strengths",
    analysis.strengths
  );


  renderList(
    "#weaknesses",
    analysis.weaknesses
  );
}


/* =========================
   RENDER LIST
========================= */

function renderList(
  selector,
  items = []
) {

  const container =
    document.querySelector(selector);

  container.innerHTML = "";


  if (!items || !items.length) {

    container.textContent =
      "No information found.";

    return;
  }


  items.forEach((item) => {

    const element =
      document.createElement("span");

    element.className =
      "skill-tag";

    element.textContent =
      item;

    container.appendChild(
      element
    );

  });
}


/* =========================
   RESUME LIBRARY
========================= */

function getSavedUser() {

  try {
    return JSON.parse(localStorage.getItem("user"));
  } catch (error) {
    localStorage.removeItem("user");
    return null;
  }
}


async function loadResumes() {

  const container = document.querySelector("#resumes-container");
  const user = getSavedUser();

  if (!container || !user?.id) {
    return;
  }

  container.innerHTML = "<p class='loading-copy'>Loading your resumes...</p>";

  try {
    const response = await fetch(
      `${API_URL}/resumes?user_id=${encodeURIComponent(user.id)}`
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(getApiError(data, "Could not load resumes."));
    }

    renderResumes(data.resumes || []);
  } catch (error) {
    console.error(error);
    container.innerHTML = `
      <div class="empty-state">
        <h2>Could not load your resumes</h2>
        <p>${escapeHtml(error.message || "The backend is unavailable.")}</p>
      </div>
    `;
  }
}


function renderResumes(resumes) {

  const container = document.querySelector("#resumes-container");

  if (!resumes.length) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">◇</span>
        <h2>No resumes yet</h2>
        <p>Upload a PDF or DOCX to create your first analyzed resume.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = resumes.map((resume) => {
    const createdAt = resume.created_at
      ? new Date(resume.created_at).toLocaleDateString()
      : "Date unavailable";

    return `
      <article class="resume-card panel">
        <div class="resume-card-icon">PDF</div>
        <div class="resume-card-body">
          <p class="eyebrow">RESUME VERSION</p>
          <h2>${escapeHtml(resume.file_name || "Untitled resume")}</h2>
          <p>${escapeHtml(resume.summary || "Resume analyzed successfully.")}</p>
          <span class="resume-date">Uploaded ${escapeHtml(createdAt)}</span>
        </div>
        <div class="resume-card-actions">
          <button class="button button-secondary" data-action="view-resume" data-resume-id="${resume.id}">
            View analysis
          </button>
          <button class="icon-button danger-button" aria-label="Delete resume" data-action="delete-resume" data-resume-id="${resume.id}">
            ×
          </button>
        </div>
      </article>
    `;
  }).join("");
}


async function deleteResume(resumeId) {

  const user = getSavedUser();

  if (!user?.id) {
    showToast("Please log in first.");
    return;
  }

  if (!window.confirm("Delete this resume and its match history?")) {
    return;
  }

  try {
    const response = await fetch(
      `${API_URL}/resume/${resumeId}?user_id=${encodeURIComponent(user.id)}`,
      { method: "DELETE" }
    );
    const data = await response.json();

    if (!response.ok) {
      showToast(getApiError(data, "Could not delete resume."));
      return;
    }

    if (localStorage.getItem("resume_id") === String(resumeId)) {
      localStorage.removeItem("resume_id");
    }

    showToast("Resume deleted successfully.");
    loadResumes();
  } catch (error) {
    console.error(error);
    showToast("Could not connect to the server.");
  }
}


/* =========================
   LOAD JOBS
========================= */

async function loadJobs() {

  const container =
    document.querySelector("#jobs-container");

  const title =
    document.querySelector("#jobs-title");


  if (!container) {
    return;
  }


  container.innerHTML =
    "<p>Loading jobs...</p>";

  title.textContent =
    "Loading jobs...";


  try {

    const response =
      await fetch(
        `${API_URL}/jobs`
      );


    const data =
      await response.json();


    if (!response.ok || data.error) {

      title.textContent =
        "Could not load jobs";

      container.innerHTML =
        "<p>Failed to load jobs.</p>";

      return;
    }


    renderJobs(
      data.jobs || []
    );


  } catch (error) {

    console.error(error);

    title.textContent =
      "Server unavailable";

    container.innerHTML = `
      <p>
        Could not connect to the backend.
      </p>
    `;
  }
}


/* =========================
   RENDER JOBS
========================= */

function renderJobs(jobs) {

  const container =
    document.querySelector(
      "#jobs-container"
    );

  const title =
    document.querySelector(
      "#jobs-title"
    );


  container.innerHTML = "";


  title.textContent =
    `${jobs.length} job${jobs.length === 1 ? "" : "s"} available`;


  if (!jobs.length) {

    container.innerHTML = `
      <div class="empty-state">

        <span class="empty-icon">
          ▱
        </span>

        <h2>
          No jobs found
        </h2>

        <p>
          Add a job or try another search.
        </p>

      </div>
    `;

    return;
  }


  jobs.forEach((job) => {

    const card =
      document.createElement("article");

    card.className =
      "job-card";

    card.dataset.jobId = job.id;
    card.tabIndex = 0;
    card.setAttribute("role", "button");


    card.innerHTML = `

      <div class="job-card-header">

        <div>

          <p class="eyebrow">
            ${escapeHtml(job.job_type || "JOB")}
          </p>

          <h3>
            ${escapeHtml(job.title)}
          </h3>

          <p class="job-company">
            ${escapeHtml(job.company)}
          </p>

        </div>


        <span class="job-location">
          ${escapeHtml(job.location || "Remote")}
        </span>

      </div>


      <p class="job-description">
        ${escapeHtml(
          job.description ||
          "No description available."
        )}
      </p>


      <div class="job-skills">

        <strong>
          Required skills:
        </strong>

        <span>
          ${escapeHtml(
            job.required_skills ||
            "Not specified"
          )}
        </span>

      </div>


      <div class="job-footer">

        <span>
          ${escapeHtml(
            job.salary ||
            "Salary not specified"
          )}
        </span>


        <button
          class="button button-primary match-job-button"
          data-job-id="${job.id}"
        >
          Match my resume
        </button>

        <button
          class="icon-button job-edit-button"
          aria-label="Edit job"
          data-action="edit-job"
          data-job-id="${job.id}"
        >
          ✎
        </button>

        <button
          class="icon-button danger-button"
          aria-label="Delete job"
          data-action="delete-job"
          data-job-id="${job.id}"
        >
          ×
        </button>

      </div>

    `;


    container.appendChild(
      card
    );

  });
}


/* =========================
   SEARCH JOBS
========================= */

async function searchJobs() {

  const searchInput =
    document.querySelector(
      "#job-search"
    );

  const query =
    searchInput.value.trim();


  /*
   * Empty search = load all jobs.
   */

  if (!query) {

    loadJobs();

    return;
  }


  const container =
    document.querySelector(
      "#jobs-container"
    );

  const title =
    document.querySelector(
      "#jobs-title"
    );


  container.innerHTML =
    "<p>Searching...</p>";

  title.textContent =
    "Searching...";


  try {

    /*
     * Search using the backend endpoint.
     *
     * We currently send the same query
     * to title, location and skill.
     */

    const url =
      `${API_URL}/jobs/search` +
      `?title=${encodeURIComponent(query)}` +
      `&location=${encodeURIComponent(query)}` +
      `&skill=${encodeURIComponent(query)}`;


    const response =
      await fetch(url);


    const data =
      await response.json();


    if (!response.ok || data.error) {

      container.innerHTML =
        "<p>Search failed.</p>";

      return;
    }


    renderJobs(
      data.jobs || []
    );


  } catch (error) {

    console.error(error);

    container.innerHTML =
      "<p>Could not connect to the server.</p>";
  }
}


/* =========================
   ESCAPE HTML
========================= */

function escapeHtml(value) {

  const div =
    document.createElement("div");

  div.textContent =
    String(value);

  return div.innerHTML;
}


/* =========================
   EVENT LISTENERS
========================= */

document.addEventListener(
  "click",
  (event) => {

    /*
     * Authentication tabs
     */

    const authTrigger =
      event.target.closest(
        "[data-auth-mode]"
      );


    if (authTrigger) {

      setAuthMode(
        authTrigger.dataset.authMode
      );

      return;
    }


    /*
     * Navigation
     */

    const navLink =
      event.target.closest(
        "[data-view]"
      );


    if (navLink) {

      event.preventDefault();

      setView(
        navLink.dataset.view
      );

      return;
    }


    /*
     * Internal view links
     */

    const viewLink =
      event.target.closest(
        "[data-view-link]"
      );


    if (viewLink) {

      setView(
        viewLink.dataset.viewLink
      );

      return;
    }


    const jobCard =
      event.target.closest(".job-card");

    if (
      jobCard &&
      !event.target.closest(".match-job-button") &&
      !event.target.closest("[data-action='edit-job']") &&
      !event.target.closest("[data-action='delete-job']")
    ) {

      matchResumeWithJob(
        jobCard.dataset.jobId
      );

      return;
    }


    /*
     * Actions
     */

    const action =
      event.target.closest(
        "[data-action]"
      )?.dataset.action;


    if (action === "menu") {

      sidebar.classList.toggle(
        "open"
      );
    }


    if (action === "upload") {

      resumeInput.click();
    }


    if (action === "advisor") {

      openAdvisor();
    }


    if (action === "close-modal") {

      closeAdvisor();
    }


    if (action === "close-job-modal") {

      closeJobModal();
    }


    if (action === "password") {

      passwordInput.type =
        passwordInput.type === "password"
          ? "text"
          : "password";


      event.target.textContent =
        passwordInput.type === "password"
          ? "Show"
          : "Hide";
    }


    if (action === "notifications") {

      showToast(
        "Notifications will appear after your first analysis."
      );
    }


    if (action === "help") {

      showToast(
        "Help center connection coming soon."
      );
    }


    if (action === "add-job") {

      openJobModal();
    }


    if (action === "edit-job") {

      openEditJobModal(
        event.target.closest("[data-job-id]").dataset.jobId
      );
    }


    if (action === "delete-job") {

      deleteJob(
        event.target.closest("[data-job-id]").dataset.jobId
      );
    }


    if (action === "delete-resume") {

      deleteResume(event.target.closest("[data-resume-id]").dataset.resumeId);
    }


    if (action === "view-resume") {

      viewResume(event.target.closest("[data-resume-id]").dataset.resumeId);
    }


    if (action === "send-advisor") {

      requestCareerAdvice();
    }


    if (action === "logout") {

      localStorage.removeItem(
        "user"
      );

      localStorage.removeItem(
        "resume_id"
      );

      appShell.classList.add(
        "hidden"
      );

      authShell.classList.remove(
        "hidden"
      );

      authForm.reset();

      setAuthMode(
        "login"
      );
    }


    /*
     * Job matching
     */

    if (
      event.target.classList.contains(
        "match-job-button"
      )
    ) {

      const jobId =
        event.target.dataset.jobId;

      matchResumeWithJob(
        jobId
      );
    }

  }
);


/* =========================
   AUTH FORM
========================= */

authForm.addEventListener(
  "submit",
  async (event) => {

    event.preventDefault();


    const formData =
      new FormData(
        authForm
      );


    const email =
      formData.get("email");

    const password =
      formData.get("password");

    const fullName =
      formData.get("name");


    try {

      const endpoint =
        authMode === "register"
          ? `${API_URL}/register`
          : `${API_URL}/login`;


      const body =
        authMode === "register"
          ? {
              full_name: fullName,
              email: email,
              password: password
            }
          : {
              email: email,
              password: password
            };


      const response =
        await fetch(
          endpoint,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json"
            },

            body:
              JSON.stringify(body)
          }
        );


      const data =
        await response.json();


      if (!response.ok) {

        showToast(
          getApiError(data, "Something went wrong.")
        );

        return;
      }


      /*
       * Register
       */

      if (authMode === "register") {

        if (!response.ok || data.error || data.message !== "User registered successfully!") {

          showToast(
            getApiError(data, "Could not create account.")
          );

          return;
        }


        showToast(
          "Account created successfully."
        );


        setAuthMode(
          "login"
        );


        document.querySelector(
          "#email"
        ).value = email;


        document.querySelector(
          "#password"
        ).value = "";


        return;
      }


      /*
       * Login
       */

      if (!response.ok || !data.user) {

        showToast(
          getApiError(data, "Invalid email or password.")
        );

        return;
      }


      if (data.user) {

        localStorage.setItem(
          "user",
          JSON.stringify(data.user)
        );


        showToast(
          "Welcome back."
        );


        showApp(
          data.user.full_name
        );

      }

    } catch (error) {

      console.error(error);

      showToast(
        "Could not connect to the server."
      );
    }
  }
);


/* =========================
   RESUME INPUT
========================= */

resumeInput.addEventListener(
  "change",
  (event) => {

    handleUpload(
      event.target.files[0]
    );

  }
);


/* =========================
   DRAG & DROP
========================= */

document
  .querySelectorAll(".empty-hero")
  .forEach((hero) => {

    ["dragenter", "dragover"]
      .forEach((eventName) => {

        hero.addEventListener(
          eventName,
          (event) => {

            event.preventDefault();

            hero.classList.add(
              "dragging"
            );

          }
        );

      });


    ["dragleave", "drop"]
      .forEach((eventName) => {

        hero.addEventListener(
          eventName,
          (event) => {

            event.preventDefault();

            hero.classList.remove(
              "dragging"
            );

          }
        );

      });


    hero.addEventListener(
      "drop",
      (event) => {

        handleUpload(
          event.dataTransfer.files[0]
        );

      }
    );

  });


/* =========================
   MODAL
========================= */

modal.addEventListener(
  "click",
  (event) => {

    if (
      event.target === modal
    ) {

      closeAdvisor();

    }

  }
);


jobModal.addEventListener(
  "click",
  (event) => {

    if (event.target === jobModal) {
      closeJobModal();
    }

  }
);


/* =========================
   KEYBOARD
========================= */

document.addEventListener(
  "keydown",
  (event) => {

    if (event.key === "Escape") {

      closeAdvisor();
      closeJobModal();

      sidebar.classList.remove(
        "open"
      );

    }


    if (
      event.key === "Enter" &&
      document.activeElement?.id === "job-search"
    ) {

      searchJobs();

    }


    if (
      event.key === "Enter" &&
      document.activeElement?.closest(".job-card") &&
      !document.activeElement?.closest(".match-job-button")
    ) {

      matchResumeWithJob(
        document.activeElement.dataset.jobId
      );

    }

  }
);


/* =========================
   JOB SEARCH BUTTON
========================= */

document
  .querySelector("#job-search-button")
  .addEventListener(
    "click",
    searchJobs
  );


jobForm.addEventListener(
  "submit",
  async (event) => {

    event.preventDefault();

    const formData = new FormData(jobForm);
    const submitButton = jobForm.querySelector("button[type='submit']");
    const editingId = jobForm.dataset.editingId;
    const body = {
      title: formData.get("title"),
      company: formData.get("company"),
      location: formData.get("location") || "Remote",
      description: formData.get("description"),
      required_skills: formData.get("required_skills") || "",
      salary: formData.get("salary") || "Not specified",
      job_type: formData.get("job_type") || "Full-time"
    };

    submitButton.disabled = true;
    submitButton.textContent = editingId
      ? "Updating job..."
      : "Analyzing job...";

    try {
      const response = await fetch(
        editingId
          ? `${API_URL}/jobs/${editingId}`
          : `${API_URL}/jobs`,
        {
        method: editingId ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
        }
      );
      const data = await response.json();

      if (!response.ok) {
        showToast(getApiError(data, "Could not add job."));
        return;
      }

      closeJobModal();
      showToast(
        editingId
          ? "Job updated successfully."
          : "Job added and analyzed successfully."
      );
      setView("jobs");
      loadJobs();
    } catch (error) {
      console.error(error);
      showToast("Could not connect to the server.");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Add job and analyze skills →";
    }
  }
);


/* =========================
   JOB MATCHING
========================= */

function displayMatchResult(data) {

  document.querySelector("#job-detail-title").textContent =
    data.job_title || "Job match";

  document.querySelector("#job-detail-meta").textContent =
    `Resume ${data.resume_id} · Job ${data.job_id}`;

  const suitabilityStatus =
    document.querySelector("#suitability-status");

  suitabilityStatus.classList.toggle(
    "not-suitable",
    !data.is_suitable
  );

  document.querySelector("#suitability-label").textContent =
    data.suitability || (
      data.is_suitable
        ? "Suitable"
        : "Needs development"
    );

  document.querySelector("#match-score").textContent =
    `${data.match_score ?? 0}%`;

  document.querySelector("#match-explanation-text").textContent =
    data.explanation || "No explanation was returned.";

  renderList("#matched-skills", data.matched_skills);
  renderList("#missing-skills", data.missing_skills);
  renderList(
    "#preferred-matched-skills",
    data.preferred_matched_skills
  );

  renderImprovementPlan(data.improvement_plan);

  document.querySelector("#job-detail-loading").hidden = true;
  document.querySelector("#job-detail-result").hidden = false;
}


function renderImprovementPlan(items = []) {

  const container = document.querySelector("#improvement-plan");
  container.innerHTML = "";

  if (!items || !items.length) {
    container.textContent = "Keep building evidence through practical projects.";
    return;
  }

  items.forEach((item) => {
    const element = document.createElement("div");
    element.className = "improvement-item";
    element.textContent = item;
    container.appendChild(element);
  });
}

async function matchResumeWithJob(
  jobId
) {

  const resumeId =
    localStorage.getItem(
      "resume_id"
    );


  if (!resumeId) {

    showToast(
      "Upload and analyze your resume first."
    );

    return;
  }

  setView("job-detail");
  document.querySelector("#job-detail-loading").hidden = false;
  document.querySelector("#job-detail-result").hidden = true;


  try {

    const response =
      await fetch(
        `${API_URL}/jobs/${jobId}/match?resume_id=${resumeId}`,
        {
          method: "POST"
        }
      );


    const data =
      await response.json();


    if (!response.ok || data.error) {

      showToast(
        getApiError(data, "Could not calculate job match.")
      );

      return;
    }


    displayMatchResult(data);


  } catch (error) {

    console.error(error);

    showToast(
      "Could not connect to the server."
    );
  }
}


/* =========================
   INITIALIZATION
========================= */

let savedUser;

try {
  savedUser = JSON.parse(localStorage.getItem("user"));
} catch (error) {
  localStorage.removeItem("user");
}

if (savedUser?.id && savedUser.full_name) {
  showApp(savedUser.full_name);
} else {
  setAuthMode("register");
}


async function viewResume(resumeId) {

  try {
    const response = await fetch(`${API_URL}/resume/${resumeId}`);
    const data = await response.json();

    if (!response.ok || data.error) {
      showToast(getApiError(data, "Could not load resume analysis."));
      return;
    }

    localStorage.setItem("resume_id", data.id);
    setView("analysis");
    displayResumeAnalysis(data);
  } catch (error) {
    console.error(error);
    showToast("Could not connect to the server.");
  }
}


function renderAdvisorList(selector, items = []) {

  const container = document.querySelector(selector);
  container.innerHTML = "";

  if (!items || !items.length) {
    container.textContent = "No recommendations yet.";
    return;
  }

  items.forEach((item) => {
    const element = document.createElement("div");
    element.className = "advisor-list-item";
    element.textContent = item;
    container.appendChild(element);
  });
}